from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import numpy as np
import tyro

from xclients.core.cfg import Config
from xclients.core.run.scene import RerunScene

logging.basicConfig(level=logging.INFO)


@dataclass
class URDFSceneConfig(Config):
    host: str = "0.0.0.0"
    port: int = 8000
    urdf: Path = Path("xarm7_standalone.urdf")
    app_id: str = "urdf_scene"
    cams: list[Path] = field(default_factory=list)
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
        self.cams = [path.expanduser().resolve() for path in self.cams]


def sample_joint_value(
    lower: float,
    upper: float,
    rng: np.random.Generator,
    mean_rad: float,
    std_rad: float,
) -> float:
    value = float(rng.normal(loc=mean_rad, scale=std_rad))

    if np.isfinite(lower):
        value = max(value, lower)
    if np.isfinite(upper):
        value = min(value, upper)

    return value


def main(cfg: URDFSceneConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    mean_rad = np.deg2rad(cfg.joint_mean_deg)
    std_rad = np.deg2rad(cfg.joint_std_deg)

    scene = RerunScene(
        cfg.urdf,
        app_id=cfg.app_id,
        camera_ht_files=cfg.cams,
        entity_path_prefix=cfg.entity_path_prefix,
        transforms_path=cfg.transforms_path,
        spawn=cfg.spawn,
        rrd_path=cfg.rrd_path,
    )

    logging.info(
        "Animating %d actuated joints from %s with mean=%.1f deg std=%.1f deg",
        len(scene.joints),
        cfg.urdf,
        cfg.joint_mean_deg,
        cfg.joint_std_deg,
    )

    for step in range(cfg.steps):
        joint_values = {
            joint.name: sample_joint_value(
                joint.limit_lower,
                joint.limit_upper,
                rng=rng,
                mean_rad=mean_rad,
                std_rad=std_rad,
            )
            for joint in scene.joints
        }
        scene.log_joints(joint_values, step=step)

        time.sleep(cfg.dt)


if __name__ == "__main__":
    main(tyro.cli(URDFSceneConfig))
