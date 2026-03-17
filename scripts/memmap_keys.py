from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich import print
import tyro

from xclients.core.cfg import spec
from xclients.io import DataLoader


@dataclass
class Config:
    path: Path
    limit: int | None = None


def main(cfg: Config) -> None:
    loader = DataLoader(cfg.path)
    seen = 0

    for stream in loader.streams:
        print(f"# {stream.dat_path} len={len(stream)}")
        for item in stream:
            print(list(item.keys()))
            print(spec(item))
            seen += 1
            if cfg.limit is not None and seen >= cfg.limit:
                return


if __name__ == "__main__":
    main(tyro.cli(Config))
