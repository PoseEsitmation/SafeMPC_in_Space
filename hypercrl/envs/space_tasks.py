"""Task families for SatDynEnv. The env, the planner cost and the CBF/CLF all read these specs."""
import dataclasses
import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np

NOMINAL_SCALE_TORQUE = 2.0
NOMINAL_U_MAX_TORQUE = NOMINAL_SCALE_TORQUE * np.sqrt(3.0)
NOMINAL_INERTIA = ((60.0, 5.0, 1.0), (5.0, 50.0, 2.0), (1.0, 2.0, 70.0))


def max_torque(B):
    """max |B u| over u in [-1, 1]^m (attained at a corner of the box)."""
    B = np.asarray(B, dtype=np.float64)
    return max(float(np.linalg.norm(B @ np.array(c)))
               for c in itertools.product((-1.0, 1.0), repeat=B.shape[1]))


@dataclass(frozen=True)
class SpaceTaskSpec:
    name: str
    angle_bound_lower: float = 80.0
    angle_bound_upper: float = 180.0
    half_angle_low_deg: float = 15.0
    half_angle_high_deg: float = 30.0
    cone_offset_deg: float = 0.0
    beta: float = 10.0
    alpha: float = 66.0
    inertia: tuple = NOMINAL_INERTIA
    scale_torque: float = NOMINAL_SCALE_TORQUE
    thruster_health: tuple = (1.0, 1.0, 1.0)
    allocation: Optional[tuple] = None   # explicit 3x3 B, overrides thruster_health

    def B(self):
        if self.allocation is not None:
            return np.array(self.allocation, dtype=np.float64)
        return np.diag(np.asarray(self.thruster_health, dtype=np.float64)) * self.scale_torque

    def env_kwargs(self):
        kwargs = dataclasses.asdict(self)
        kwargs.pop("name")
        return kwargs

    def summary(self):
        """Name plus every field that differs from the nominal task (for tasks.json)."""
        nominal = SpaceTaskSpec("nominal")
        d = {"name": self.name}
        d.update({k: v for k, v in dataclasses.asdict(self).items()
                  if k != "name" and v != getattr(nominal, k)})
        d["u_max_torque"] = round(max_torque(self.B()), 6)
        return d


# Scenario and reward vary, dynamics are identical.
DIFFICULTY = [
    SpaceTaskSpec("default"),
    SpaceTaskSpec("easy", angle_bound_lower=10.0, angle_bound_upper=45.0),
    SpaceTaskSpec("hard", angle_bound_lower=90.0, beta=50.0, alpha=100.0),
    SpaceTaskSpec("tight_koz", half_angle_low_deg=25.0, half_angle_high_deg=40.0),
]

MOI = [
    SpaceTaskSpec("asymmetric"),
    SpaceTaskSpec("small_symmetric", inertia=((20.0, 1.0, 0.0), (1.0, 22.0, 0.0), (0.0, 0.0, 25.0))),
    SpaceTaskSpec("large_heavy", inertia=((120.0, 10.0, 3.0), (10.0, 90.0, 5.0), (3.0, 5.0, 150.0))),
    SpaceTaskSpec("oblate", inertia=((80.0, 2.0, 0.0), (2.0, 80.0, 0.0), (0.0, 0.0, 20.0))),
]

# Faults bottom out at 0.15, not 0: two-torque attitude control is not stabilizable
# by continuous static feedback, so a dead axis would be unlearnable for the policy.
THRUSTER = [
    SpaceTaskSpec("nominal"),
    SpaceTaskSpec("roll_degraded", thruster_health=(0.35, 1.0, 1.0)),
    SpaceTaskSpec("yaw_near_dead", thruster_health=(1.0, 1.0, 0.15)),
    SpaceTaskSpec("double_fault", thruster_health=(0.5, 0.15, 1.0)),
    SpaceTaskSpec("misaligned", allocation=((1.90, 0.50, 0.00),
                                            (-0.50, 1.80, 0.40),
                                            (0.10, -0.40, 1.70))),
]

# Negative control: identical tasks.
NULL = [SpaceTaskSpec(f"identical_{i}") for i in range(4)]

FAMILIES = {
    "spaceEnv": DIFFICULTY,
    "spaceEnv_moi": MOI,
    "spaceEnv_thruster": THRUSTER,
    "spaceEnv_null": NULL,
}

_scenario_override = {}


def set_scenario_override(**fields):
    """Pin scenario geometry for every task of the run (--fixed-scenario)."""
    _scenario_override.clear()
    _scenario_override.update(fields)


def get_task_spec(env_name, task_id):
    family = FAMILIES[env_name]
    spec = family[min(task_id, len(family) - 1)]
    return dataclasses.replace(spec, **_scenario_override) if _scenario_override else spec
