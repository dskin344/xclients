from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    nbytes: int

    @classmethod
    def from_raw(cls, name: str, raw: Any) -> FieldSpec:
        if isinstance(raw, int):
            shape = (raw,)
            dtype = np.dtype(np.float32)
        elif isinstance(raw, list):
            shape = tuple(int(dim) for dim in raw)
            dtype = np.dtype(np.uint8)
        elif isinstance(raw, dict):
            shape_raw = raw.get("shape", raw.get("dims"))
            if shape_raw is None:
                raise ValueError(f"Field {name!r} is missing a shape in the JSON schema.")
            shape = (int(shape_raw),) if isinstance(shape_raw, int) else tuple(int(dim) for dim in shape_raw)
            dtype = np.dtype(raw.get("dtype", np.float32 if len(shape) == 1 else np.uint8))
        else:
            raise TypeError(f"Unsupported schema entry for {name!r}: {type(raw)!r}")

        item_count = int(np.prod(shape, dtype=np.int64))
        return cls(name=name, shape=shape, dtype=dtype, nbytes=item_count * dtype.itemsize)


@dataclass(frozen=True)
class Metadata:
    path: Path
    schema: dict[str, Any]
    info: dict[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> Metadata:
        payload = json.loads(path.read_text())
        schema = payload.get("schema")
        if not isinstance(schema, dict):
            raise ValueError(f"{path} is missing a top-level 'schema' object.")
        info = payload.get("info", {})
        if not isinstance(info, dict):
            raise ValueError(f"{path} has a non-dict 'info' payload.")
        return cls(path=path, schema=schema, info=info)

    @property
    def field_specs(self) -> tuple[FieldSpec, ...]:
        return tuple(FieldSpec.from_raw(name, raw) for name, raw in self.schema.items())

    @property
    def payload_nbytes(self) -> int:
        return sum(field.nbytes for field in self.field_specs)

    @property
    def declared_length(self) -> int | None:
        value = self.info.get("len")
        return int(value) if value is not None else None

    @property
    def declared_capacity(self) -> int | None:
        value = self.info.get("maxlen")
        return int(value) if value is not None else None

    @property
    def referenced_dat_name(self) -> str | None:
        value = self.info.get("path")
        if not value:
            return None
        return Path(str(value)).name


@dataclass(frozen=True)
class MetadataMatch:
    metadata: Metadata
    kind: str


class MemmapSequence:
    def __init__(self, dat_path: Path, match: MetadataMatch):
        self.dat_path = dat_path
        self.metadata = match.metadata
        self.match_kind = match.kind
        self.fields = self.metadata.field_specs
        self.payload_nbytes = self.metadata.payload_nbytes
        self.row_nbytes = self._infer_row_nbytes()
        self.extra_nbytes = self.row_nbytes - self.payload_nbytes
        self._buf = np.memmap(self.dat_path, mode="r", dtype=np.uint8)
        if self._buf.size % self.row_nbytes:
            raise ValueError(f"{self.dat_path} size {self._buf.size} is not divisible by row size {self.row_nbytes}.")
        self.row_count = self._buf.size // self.row_nbytes
        self._rows = self._buf.reshape(self.row_count, self.row_nbytes)
        self.length = self._infer_length()

    def _infer_row_nbytes(self) -> int:
        file_size = self.dat_path.stat().st_size
        candidates: list[int] = []
        for count in (self.metadata.declared_capacity, self.metadata.declared_length):
            if count and count > 0 and file_size % count == 0:
                row_nbytes = file_size // count
                if row_nbytes >= self.payload_nbytes:
                    candidates.append(row_nbytes)
        if file_size % self.payload_nbytes == 0:
            candidates.append(self.payload_nbytes)
        if not candidates:
            raise ValueError(f"Could not infer a row size for {self.dat_path} from {self.metadata.path}.")
        return min(candidates)

    def _infer_length(self) -> int:
        if self.match_kind.startswith("exact"):
            declared = self.metadata.declared_length
            if declared is not None:
                return min(declared, self.row_count)
            inferred = self._infer_nonzero_length()
            if inferred:
                return inferred
            return self.row_count

        inferred = self._infer_nonzero_length()
        if inferred:
            return inferred
        return 0

    def _infer_nonzero_length(self, chunk_size: int = 32) -> int:
        for end in range(self.row_count, 0, -chunk_size):
            start = max(0, end - chunk_size)
            block = self._rows[start:end]
            mask = np.any(block != 0, axis=1)
            if mask.any():
                return start + int(np.flatnonzero(mask)[-1]) + 1
        return 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[dict[str, np.ndarray | float]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> dict[str, np.ndarray | float]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        row = self._rows[index]
        item: dict[str, np.ndarray | float] = {}
        offset = 0

        if self.extra_nbytes == 4:
            item["time"] = float(np.frombuffer(row[:4], dtype=np.float32)[0])
            offset = 4
        elif self.extra_nbytes:
            item["_prefix"] = row[: self.extra_nbytes].copy()
            offset = self.extra_nbytes

        for field in self.fields:
            chunk = row[offset : offset + field.nbytes]
            item[field.name] = np.frombuffer(chunk, dtype=field.dtype).reshape(field.shape)
            offset += field.nbytes

        return item

    def as_dict_of_sequences(self) -> dict[str, np.ndarray]:
        sequences: dict[str, np.ndarray] = {}
        if self.extra_nbytes == 4:
            sequences["time"] = np.frombuffer(
                self._rows[: len(self), :4].copy().reshape(-1),
                dtype=np.float32,
            )
        elif self.extra_nbytes:
            sequences["_prefix"] = self._rows[: len(self), : self.extra_nbytes].copy()

        offset = self.extra_nbytes
        for field in self.fields:
            chunk = self._rows[: len(self), offset : offset + field.nbytes]
            sequences[field.name] = chunk.view(field.dtype).reshape(len(self), *field.shape)
            offset += field.nbytes

        return sequences


class DataLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)

        self.streams = self._discover_streams()
        if not self.streams:
            raise FileNotFoundError(f"No readable memmap streams found under {self.path}.")

    def _discover_streams(self) -> list[MemmapSequence]:
        dat_paths = sorted(self.path.glob("*.dat"))
        metadata = [Metadata.from_path(path) for path in sorted(self.path.glob("*.json"))]

        streams: list[MemmapSequence] = []
        for dat_path in dat_paths:
            match = self._match_metadata(dat_path, metadata)
            if match is None:
                log.warning("Skipping %s: no compatible JSON schema found.", dat_path.name)
                continue
            try:
                streams.append(MemmapSequence(dat_path, match))
            except ValueError as exc:
                log.warning("Skipping %s: %s", dat_path.name, exc)

        return streams

    def _match_metadata(self, dat_path: Path, metadata: list[Metadata]) -> MetadataMatch | None:
        for meta in metadata:
            if meta.referenced_dat_name == dat_path.name:
                return MetadataMatch(meta, "exact_info")
            if meta.path.with_suffix(".dat").name == dat_path.name:
                return MetadataMatch(meta, "exact_stem")

        compatible = [meta for meta in metadata if self._is_compatible(dat_path, meta)]
        if not compatible:
            return None

        dat_stamp = _timestamp_value(dat_path)
        if dat_stamp is None:
            return MetadataMatch(compatible[0], "compatible")

        closest = min(
            compatible,
            key=lambda meta: abs((_timestamp_value(meta.path) or dat_stamp) - dat_stamp),
        )
        return MetadataMatch(closest, "nearest_compatible")

    @staticmethod
    def _is_compatible(dat_path: Path, meta: Metadata) -> bool:
        file_size = dat_path.stat().st_size
        for count in (meta.declared_capacity, meta.declared_length):
            if count and count > 0 and file_size % count == 0:
                return (file_size // count) >= meta.payload_nbytes
        return file_size % meta.payload_nbytes == 0

    def __len__(self) -> int:
        return sum(len(stream) for stream in self.streams)

    def __iter__(self) -> Iterator[dict[str, np.ndarray | float]]:
        for stream in self.streams:
            yield from stream


def _timestamp_value(path: Path) -> int | None:
    match = re.search(r"(\d{8})-(\d{6})", path.stem)
    if match is None:
        return None
    return int("".join(match.groups()))
