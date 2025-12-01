"""
Shared control signal utilities (thruster command sampler).
"""
from __future__ import annotations

import numpy as np


def command_sampler(
    mode: str,
    vehicle,
    rng: np.random.Generator,
    step: int,
    dt: float,
    period: int,
    fwd: float,
    diff: float,
    omega: float,
) -> np.ndarray:
    """
    Generate propeller rev commands [n1, n2] (rad/s) for a given pattern.
    The sampler matches training/eval/collection so datasets and rollouts stay consistent.
    """
    n_min = float(vehicle.n_min)
    n_max = float(vehicle.n_max)

    def scale_to_revs(bias_frac: float, delta_frac: float) -> np.ndarray:
        bias = bias_frac * n_max
        delta = delta_frac * n_max
        return np.clip(np.array([bias + delta, bias - delta]), n_min, n_max)

    if mode == "random":
        mag = rng.uniform(0.2, 0.9) * n_max
        delta = rng.uniform(-0.5, 0.5) * n_max
        if rng.random() < 0.2:
            mag = -mag
        cmd = np.clip(np.array([mag + delta, mag - delta]), n_min, n_max)
    elif mode == "zigzag":
        sign = 1.0 if (step // period) % 2 == 0 else -1.0
        delta_frac = diff * sign
        cmd = scale_to_revs(fwd, delta_frac)
    elif mode == "figure8":
        t = step * dt
        delta_frac = diff * np.sin(omega * t) * np.cos(omega * t)
        cmd = scale_to_revs(fwd, delta_frac)
    elif mode == "circle":
        t = step * dt
        delta_frac = diff * np.sin(omega * t)
        cmd = scale_to_revs(fwd, delta_frac)
    else:
        raise ValueError(f"Unknown pattern {mode}")

    return cmd

