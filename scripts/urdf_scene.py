from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time

import numpy as np
import rerun as rr
import rerun.urdf as rru
import tyro

from xclients.core.cfg import Config

logging.basicConfig(level=logging.INFO)


@dataclass
class URDFSceneConfig(Config):
    host: str = "0.0.0.0"
    port: int = 8000
    urdf: Path = Path("xarm7_standalone.urdf")
    app_id: str = "urdf_scene"
    entity_path_prefix: str = "robot"
    transforms_path: str = "robot/transforms"
    spawn: bool = True
    rrd_path: Path | None = Path("urdf_scene.rrd")
    steps: int = 200
    dt: float = 0.05
    seed: int = 0
    joint_mean_deg: float = 0.0
    joint_std_deg: float = 90.0

    def __post_init__(self) -> None:
        self.urdf = self.urdf.expanduser().resolve()


def sample_joint_value(joint: rru.UrdfJoint, rng: np.random.Generator, mean_rad: float, std_rad: float) -> float:
    value = float(rng.normal(loc=mean_rad, scale=std_rad))

    if np.isfinite(joint.limit_lower):
        value = max(value, joint.limit_lower)
    if np.isfinite(joint.limit_upper):
        value = min(value, joint.limit_upper)

    return value


def batch_joint_transforms(joints: list[rru.UrdfJoint], joint_values: list[float]) -> rr.ComponentColumnList:
    translations = []
    quaternions = []
    parent_frames = []
    child_frames = []

    for joint, joint_value in zip(joints, joint_values, strict=True):
        transform = joint.compute_transform(joint_value, clamp=True)
        translations.append(transform.translation.as_arrow_array().to_pylist()[0])
        quaternions.append(transform.quaternion.as_arrow_array().to_pylist()[0])
        parent_frames.append(transform.parent_frame.as_arrow_array().to_pylist()[0])
        child_frames.append(transform.child_frame.as_arrow_array().to_pylist()[0])

    return rr.Transform3D.columns(
        translation=translations,
        quaternion=quaternions,
        parent_frame=parent_frames,
        child_frame=child_frames,
    )


def main(cfg: URDFSceneConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    mean_rad = np.deg2rad(cfg.joint_mean_deg)
    std_rad = np.deg2rad(cfg.joint_std_deg)

    rr.init(cfg.app_id)
    if cfg.rrd_path is not None:
        rr.save(cfg.rrd_path)
    if cfg.spawn:
        try:
            rr.spawn()
        except RuntimeError as exc:
            logging.warning("Failed to spawn rerun viewer: %s", exc)
    rr.log("/", rr.ViewCoordinates.FLU, static=True)
    rr.log_file_from_path(cfg.urdf, entity_path_prefix=cfg.entity_path_prefix, static=True)

    urdf_tree = rru.UrdfTree.from_file_path(cfg.urdf, entity_path_prefix=cfg.entity_path_prefix)
    joints = [joint for joint in urdf_tree.joints() if joint.joint_type in {"revolute", "continuous", "prismatic"}]

    logging.info(
        "Animating %d actuated joints from %s with mean=%.1f deg std=%.1f deg",
        len(joints),
        cfg.urdf,
        cfg.joint_mean_deg,
        cfg.joint_std_deg,
    )

    for step in range(cfg.steps):
        joint_values = [sample_joint_value(joint, rng=rng, mean_rad=mean_rad, std_rad=std_rad) for joint in joints]

        rr.send_columns(
            cfg.transforms_path,
            indexes=[rr.TimeColumn("step", sequence=[step] * len(joints))],
            columns=batch_joint_transforms(joints=joints, joint_values=joint_values),
        )

        rr.set_time("step", sequence=step)
        for joint, joint_value in zip(joints, joint_values, strict=True):
            rr.log(f"robot/joint_values/{joint.name}", rr.Scalars([float(np.rad2deg(joint_value))]))

        time.sleep(cfg.dt)


if __name__ == "__main__":
    main(tyro.cli(URDFSceneConfig))
