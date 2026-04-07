from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import tyro

import xclients
from xclients.core import tf as xctf
from xclients.core.run.blueprint import init_blueprint
from xclients.core.run.fustrum import log_fustrum
from xclients.core.run.rerun_urdf import ez_load_urdf

mano_joint_pairs = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

_BONE_COLORS = {
    "wrist":   [1.0, 1.0, 0.0],
    "mcp_pip": [1.0, 0.0, 0.0],
    "pip_dip": [0.0, 1.0, 0.0],
    "dip_tip": [0.0, 0.0, 1.0],
}


def _bone_color(j1: int, j2: int) -> list[float]:
    if j1 == 0:
        return _BONE_COLORS["wrist"]
    if j2 % 4 == 0:
        return _BONE_COLORS["mcp_pip"]
    if j2 % 4 == 3:
        return _BONE_COLORS["dip_tip"]
    return _BONE_COLORS["pip_dip"]


def load_extr(name: str) -> np.ndarray:
    p = xclients.ROOT / "data/cam" / name / "HT.npz"
    data = np.load(p)
    ht = {k: data[k] for k in data.files}["HT"]
    return xctf.RDF2FLU @ np.linalg.inv(ht)


@dataclass
class Config:
    dir: Path
    urdf: Path = Path("../xarm7_standalone.urdf")
    cam_names: list[str] = field(default_factory=lambda: ["agent1", "agent2"])
    app_id: str = "view_k3ds"
    web_port: int = 9091
    grpc_port: int = 9877

    def __post_init__(self) -> None:
        self.dir = Path(self.dir).expanduser().resolve()
        if self.urdf is not None:
            self.urdf = Path(self.urdf).expanduser().resolve()
            print(self.urdf)


def log_episode(
    path: Path,
    cameras: dict,
    frame_offset: int = 0,
) -> int:
    data = dict(np.load(path).items())
    k3ds = data["k3ds"]  # [T, 21, 4]
    cam_keys = [k for k in cameras if k in data]

    for i, kp in enumerate(k3ds):
        rr.set_time("frame", sequence=frame_offset + i)

        for k in cam_keys:
            rr.log(
                f"world/cam/{k}/image",
                rr.Image(data[k][i], color_model="BGR").compress(jpeg_quality=75),
            )

        rr.log("world/kp3d", rr.Points3D(kp[:, :3], radii=np.ones(21) * 0.0025))

        segments = np.array([[kp[j1, :3], kp[j2, :3]] for j1, j2 in mano_joint_pairs])
        colors = np.array([_bone_color(j1, j2) for j1, j2 in mano_joint_pairs])
        rr.log("world/kp3d/lines", rr.LineStrips3D(segments, colors=colors))

    return len(k3ds)


def main(cfg: Config) -> None:
    files = sorted(cfg.dir.glob("processed_ep*.npz"))
    if not files:
        raise FileNotFoundError(f"No processed_ep*.npz files found in {cfg.dir}")

    h, w = 480, 640
    fx, fy = 515.0, 515.0
    intr = np.array([[fx, 0.0, w / 2], [0.0, fy, h / 2], [0.0, 0.0, 1.0]])
    cameras = {
        name: {"intrinsics": intr, "extrinsics": load_extr(name), "height": h, "width": w}
        for name in cfg.cam_names
    }

    rr.init(cfg.app_id, spawn=False)
    grpc_url = rr.serve_grpc(grpc_port=cfg.grpc_port)
    rr.serve_web_viewer(web_port=cfg.web_port, open_browser=False, connect_to=grpc_url)
    print(f"grpc_url: {grpc_url}")
    print(f"Open http://localhost:{cfg.web_port}?url={grpc_url} in your browser.")
    input("Press Enter when ready to load data...")
    rr.log("/", rr.ViewCoordinates.FLU, static=True)

    init_blueprint(cfg.cam_names)
    log_fustrum(cameras, root=Path("world"))

    if cfg.urdf is not None:
        ez_load_urdf(cfg.urdf)
        rr.log("/robot/urdf", rr.ViewCoordinates.FLU, static=True)

    offset = 0
    for path in files:
        offset += log_episode(path, cameras, frame_offset=offset)

    input("Press Enter to exit...")


if __name__ == "__main__":
    main(tyro.cli(Config))
