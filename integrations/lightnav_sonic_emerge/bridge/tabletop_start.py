"""Reproducible runtime starts, independent of navigation and scene files."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TableFootprint:
    centre_x: float
    centre_y: float
    half_x: float
    half_y: float
    yaw: float = 0.0

    def local_xy(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self.centre_x, y - self.centre_y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def contains(self, x: float, y: float, padding: float = 0.0) -> bool:
        lx, ly = self.local_xy(x, y)
        return abs(lx) <= self.half_x + padding and abs(
            ly) <= self.half_y + padding

    def as_dict(self) -> dict:
        return vars(self).copy()


def table_footprint(model, data) -> TableFootprint:
    """Read the highest box on the existing table body (no scene mutation)."""
    import mujoco

    table_id = model.body("tabletop_lab_table").id
    candidates = [
        i for i in range(model.ngeom)
        if int(model.geom_bodyid[i]) == table_id and
        int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    ]
    if not candidates:
        raise ValueError("scene has no box tabletop")
    i = max(candidates, key=lambda i: float(data.geom_xpos[i, 2]))
    rotation = data.geom_xmat[i].reshape(3, 3)
    return TableFootprint(
        float(data.geom_xpos[i, 0]), float(data.geom_xpos[i, 1]),
        float(model.geom_size[i, 0]), float(model.geom_size[i, 1]),
        math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))


class RandomStartSampler:
    """Uniform disk sampling, conditioned on table clearance; random world yaw."""

    def __init__(self,
                 table: TableFootprint,
                 seed: int | None = None,
                 clearance_m: float = 0.6) -> None:
        self.table = table
        self.seed = random.SystemRandom().getrandbits(
            63) if seed is None else seed
        self.random = random.Random(self.seed)
        self.clearance_m = clearance_m
        self.index = 0

    def sample(self, radius_m: float, sample_seed: int | None = None) -> dict:
        if not math.isfinite(radius_m) or radius_m <= min(
                self.table.half_x, self.table.half_y) + self.clearance_m:
            raise ValueError(
                "radius must exceed the table half-width plus 0.6 m clearance")
        seed = self.random.getrandbits(
            63) if sample_seed is None else sample_seed
        rng = random.Random(seed)
        for _ in range(10000):
            r, angle = radius_m * math.sqrt(rng.random()), rng.uniform(
                -math.pi, math.pi)
            x = self.table.centre_x + r * math.cos(angle)
            y = self.table.centre_y + r * math.sin(angle)
            if not self.table.contains(x, y, self.clearance_m):
                self.index += 1
                return {
                    "mode": "random",
                    "radius_m": radius_m,
                    "base_seed": self.seed,
                    "sample_seed": seed,
                    "sample_index": self.index,
                    "clearance_m": self.clearance_m,
                    "table": self.table.as_dict(),
                    "position_xy_m": [x, y],
                    "yaw_rad": rng.uniform(-math.pi, math.pi)
                }
        raise ValueError(
            "radius leaves too little free area; choose a larger radius")


def apply_start(qpos, qpos_adr: int, config: dict) -> None:
    """Change only root XY/yaw, retaining initial Z, roll/pitch, and joint angles."""
    w, x, y, z = (float(v) for v in qpos[qpos_adr + 3:qpos_adr + 7])
    old_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    delta = (config["yaw_rad"] - old_yaw) / 2
    c, s = math.cos(delta), math.sin(delta)
    qpos[qpos_adr:qpos_adr + 2] = config["position_xy_m"]
    qpos[qpos_adr + 3:qpos_adr +
         7] = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
