"""
Collect excitation rollouts from the Otter simulator.

Usage:
    python -m asv_framework.data.collect_otter_rollouts --episodes 5 --steps 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from asv_framework.otter_env import OtterEnv


def thrust_limits(vehicle):
    # Maximum thrust from bollard curves
    pos = vehicle.k_pos * vehicle.n_max ** 2
    neg = -vehicle.k_neg * vehicle.n_min ** 2  # n_min is negative
    return pos, -neg  # return positive bounds: (pos_max, neg_min_abs)


def command_sampler(mode, vehicle, rng, step, dt, period, fwd, diff, omega, thrust_mode):
    """
    Returns either rev commands [n1,n2] or thrust commands [F1,F2] depending on thrust_mode.
    """
    if thrust_mode:
        pos_max, neg_max = thrust_limits(vehicle)
        def scale_to_thrust(bias_frac, delta_frac):
            bias = bias_frac * pos_max
            delta = delta_frac * pos_max
            return np.clip(np.array([bias + delta, bias - delta]), -neg_max, pos_max)
    else:
        n_min = float(vehicle.n_min); n_max = float(vehicle.n_max)
        def scale_to_revs(bias_frac, delta_frac):
            bias = bias_frac * n_max
            delta = delta_frac * n_max
            return np.clip(np.array([bias + delta, bias - delta]), n_min, n_max)

    if mode == "random":
        if thrust_mode:
            mag = rng.uniform(0.2, 1.0)
            delta_frac = rng.uniform(-0.5, 0.5)
            if rng.random() < 0.2:
                mag = -mag
            cmd = scale_to_thrust(mag, delta_frac)
        else:
            n_min = float(vehicle.n_min); n_max = float(vehicle.n_max)
            mag = rng.uniform(0.2, 1.0) * n_max
            delta = rng.uniform(-0.5, 0.5) * n_max
            if rng.random() < 0.2:
                mag = -mag
            cmd = np.clip(np.array([mag + delta, mag - delta]), n_min, n_max)
    elif mode == "zigzag":
        sign = 1.0 if (step // period) % 2 == 0 else -1.0
        delta_frac = diff * sign
        if thrust_mode:
            cmd = scale_to_thrust(fwd, delta_frac)
        else:
            cmd = scale_to_revs(fwd, delta_frac)
    elif mode == "figure8":
        t = step * dt
        delta_frac = diff * np.sin(omega * t) * np.cos(omega * t)
        if thrust_mode:
            cmd = scale_to_thrust(fwd, delta_frac)
        else:
            cmd = scale_to_revs(fwd, delta_frac)
    elif mode == "circle":
        # Constant-turn surrogate: slow sinusoidal yaw bias
        t = step * dt
        delta_frac = diff * np.sin(omega * t)
        if thrust_mode:
            cmd = scale_to_thrust(fwd, delta_frac)
        else:
            cmd = scale_to_revs(fwd, delta_frac)
    else:
        raise ValueError(f"Unknown pattern {mode}")
    return cmd


def collect_episode(
    steps: int,
    sample_time: float,
    rng: np.random.Generator,
    pattern: str,
    period: int,
    fwd: float,
    diff: float,
    omega: float,
    thrust_commands: bool,
) -> dict:
    env = OtterEnv(sample_time=sample_time, use_autopilot=False, thrust_commands=thrust_commands)
    env.reset()
    vehicle = env.vehicle

    t_list, eta_list, nu_list, u_list, u_actual_list = [], [], [], [], []
    planar_q, planar_q_dot, planar_u = [], [], []

    for k in range(steps):
        cmd = command_sampler(
            pattern,
            vehicle,
            rng,
            k,
            sample_time,
            period,
            fwd,
            diff,
            omega,
            thrust_commands,
        )

        sample = env.step(cmd)
        q, q_dot, u = env.to_planar(sample)
        t_list.append(sample["t"])
        eta_list.append(sample["eta"])
        nu_list.append(sample["nu"])
        u_list.append(sample["u_control"])
        u_actual_list.append(sample["u_actual"])
        planar_q.append(q)
        planar_q_dot.append(q_dot)
        planar_u.append(u)

    return {
        "t": np.asarray(t_list),
        "eta": np.asarray(eta_list),
        "nu": np.asarray(nu_list),
        "u_control": np.asarray(u_list),
        "u_actual": np.asarray(u_actual_list),
        "q": np.asarray(planar_q),
        "q_dot": np.asarray(planar_q_dot),
        "u": np.asarray(planar_u),
        "dt": sample_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="number of rollouts")
    parser.add_argument("--steps", type=int, default=1200, help="steps per rollout")
    parser.add_argument("--dt", type=float, default=0.05, help="sample time [s]")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "otter_rollouts.npz"),
        help="output npz file",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--pattern",
        type=str,
        default="random",
        choices=["random", "zigzag", "figure8", "circle"],
        help="thrust excitation pattern",
    )
    parser.add_argument("--period", type=int, default=150, help="zigzag period (steps)")
    parser.add_argument("--fwd", type=float, default=0.7, help="forward thrust fraction of n_max")
    parser.add_argument("--diff", type=float, default=0.4, help="thrust differential fraction of n_max")
    parser.add_argument("--omega", type=float, default=0.2, help="figure8 frequency (rad/s)")
    parser.add_argument("--thrust-commands", action="store_true", help="interpret pattern outputs as thrust [N] instead of revs")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    episodes = []
    for _ in range(args.episodes):
        episodes.append(
            collect_episode(
                args.steps,
                args.dt,
                rng,
                pattern=args.pattern,
                period=args.period,
                fwd=args.fwd,
                diff=args.diff,
                omega=args.omega,
                thrust_commands=args.thrust_commands,
            )
        )

    # Stack episodes along first dim
    data = {k: np.stack([ep[k] for ep in episodes], axis=0) for k in episodes[0]}
    np.savez(args.output, **data)
    print(f"Saved rollouts to {args.output}")


if __name__ == "__main__":
    main()
