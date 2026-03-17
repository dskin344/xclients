from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import numpy as np
import rerun as rr
import tyro

from xclients.core.run.scene import RerunScene
from xclients.io import DataLoader

logging.basicConfig(level=logging.INFO)

CAMERA_KEYS = {
    "low": "/xgym/camera/low",
    "side": "/xgym/camera/side",
}
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINT_NAMES = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


@dataclass
class Config:
    path: Path
    urdf: Path = Path("xarm7_standalone.urdf")
    app_id: str = "memmap_view"
    cams: list[Path] = field(default_factory=list)
    entity_path_prefix: str = "robot"
    transforms_path: str = "robot/transforms"
    spawn: bool = True
    rrd_path: Path | None = None
    stream: int | None = None
    limit: int | None = None
    realtime: bool = True
    dt: float | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        self.urdf = Path(self.urdf).expanduser().resolve()
        self.cams = [Path(path).expanduser().resolve() for path in self.cams]
        if self.rrd_path is not None:
            self.rrd_path = Path(self.rrd_path).expanduser().resolve()


def selected_streams(loader: DataLoader, stream_index: int | None):
    if stream_index is None:
        yield from enumerate(loader.streams)
        return

    if stream_index < 0 or stream_index >= len(loader.streams):
        raise IndexError(f"stream index {stream_index} is out of range for {len(loader.streams)} streams")
    yield stream_index, loader.streams[stream_index]


def main(cfg: Config) -> None:
    loader = DataLoader(cfg.path)
    scene = RerunScene(
        cfg.urdf,
        app_id=cfg.app_id,
        camera_ht_files=cfg.cams,
        entity_path_prefix=cfg.entity_path_prefix,
        transforms_path=cfg.transforms_path,
        spawn=cfg.spawn,
        rrd_path=cfg.rrd_path,
    )

    if not scene.camera_calibrations:
        scene.set_cameras(list(CAMERA_KEYS))

    step = 0
    for stream_index, stream in selected_streams(loader, cfg.stream):
        logging.info("Viewing stream %d from %s with %d items", stream_index, stream.dat_path, len(stream))
        last_time: float | None = None

        for item in stream:
            if cfg.limit is not None and step >= cfg.limit:
                return

            record_time = item.get("time")
            if isinstance(record_time, float):
                rr.set_time("time", duration=record_time)
                if cfg.realtime and cfg.dt is None and last_time is not None:
                    time.sleep(max(record_time - last_time, 0.0))
                last_time = record_time
            elif cfg.realtime and cfg.dt is not None:
                time.sleep(cfg.dt)

            rr.set_time("step", sequence=step)

            frames = {}
            for camera_name, key in CAMERA_KEYS.items():
                frame = item.get(key)
                if isinstance(frame, np.ndarray):
                    frames[camera_name] = frame
            if frames:
                scene.log_camera_images(frames)

            arm_joints = item.get("xarm_joints")
            if arm_joints is None:
                raise KeyError("Expected key 'xarm_joints' in memmap record.")
            arm_values = np.asarray(arm_joints, dtype=float).reshape(-1)
            if len(arm_values) != len(ARM_JOINT_NAMES):
                raise ValueError(f"Expected {len(ARM_JOINT_NAMES)} xarm joints, got {len(arm_values)}")

            joint_values = {
                name: float(value)
                for name, value in zip(ARM_JOINT_NAMES, arm_values, strict=True)
                if name in scene.joint_map
            }

            gripper = item.get("xarm_gripper")
            if gripper is not None:
                gripper_value = float(np.asarray(gripper, dtype=float).reshape(-1)[0])
                for name in GRIPPER_JOINT_NAMES:
                    if name in scene.joint_map:
                        joint_values[name] = gripper_value

            scene.log_joints(joint_values, step=step)
            step += 1


if __name__ == "__main__":
    main(tyro.cli(Config))
