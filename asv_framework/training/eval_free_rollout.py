"""
Free-rollout evaluation of the learned SE(2) HamNODE against the Otter simulator.

Compares multi-step trajectories under scripted thrust patterns (random, zigzag, figure8)
and reports RMSE over the horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from asv_framework.otter_env import OtterEnv
from asv_framework.training.train_otter_se2 import yaw_to_R_flat, wrap_angle
from asv_framework.models.se2_hamnode import SE2HamNODE


def thrust_limits(vehicle):
    pos = vehicle.k_pos * vehicle.n_max ** 2
    neg = -vehicle.k_neg * vehicle.n_min ** 2
    return pos, -neg


def command_sampler(mode, vehicle, rng, step, dt, period, fwd, diff, omega, thrust_mode):
    if thrust_mode:
        pos_max, neg_max = thrust_limits(vehicle)
        def scale_to_thrust(bias_frac, delta_frac):
            bias = bias_frac * pos_max
            delta = delta_frac * pos_max
            return np.clip(np.array([bias + delta, bias - delta]), -neg_max, pos_max)
    else:
        n_min, n_max = float(vehicle.n_min), float(vehicle.n_max)
        def scale_to_revs(bias_frac, delta_frac):
            bias = bias_frac * n_max
            delta = delta_frac * n_max
            return np.clip(np.array([bias + delta, bias - delta]), n_min, n_max)

    if mode == "random":
        if thrust_mode:
            mag = rng.uniform(0.2, 0.9)
            delta_frac = rng.uniform(-0.5, 0.5)
            if rng.random() < 0.2:
                mag = -mag
            cmd = scale_to_thrust(mag, delta_frac)
        else:
            n_min = float(vehicle.n_min); n_max = float(vehicle.n_max)
            mag = rng.uniform(0.2, 0.9) * n_max
            delta = rng.uniform(-0.5, 0.5) * n_max
            if rng.random() < 0.2:
                mag = -mag
            cmd = np.clip(np.array([mag + delta, mag - delta]), n_min, n_max)
    elif mode == "zigzag":
        sign = 1.0 if (step // period) % 2 == 0 else -1.0
        delta_frac = diff * sign
        cmd = scale_to_thrust(fwd, delta_frac) if thrust_mode else scale_to_revs(fwd, delta_frac)
    elif mode == "figure8":
        t = step * dt
        delta_frac = diff * np.sin(omega * t) * np.cos(omega * t)
        cmd = scale_to_thrust(fwd, delta_frac) if thrust_mode else scale_to_revs(fwd, delta_frac)
    elif mode == "circle":
        t = step * dt
        delta_frac = diff * np.sin(omega * t)
        cmd = scale_to_thrust(fwd, delta_frac) if thrust_mode else scale_to_revs(fwd, delta_frac)
    else:
        raise ValueError(f"Unknown pattern {mode}")
    return cmd


def rollout(args):
    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    env = OtterEnv(sample_time=args.dt, use_autopilot=False, thrust_commands=args.thrust_commands)
    env.reset()
    rng = np.random.default_rng(args.seed)

    model = SE2HamNODE(device=device).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Initial predicted state
    q_pred = torch.zeros(3, dtype=torch.float32, device=device)
    qd_pred = torch.zeros(3, dtype=torch.float32, device=device)

    qs_true, qs_pred = [], []
    qd_true_list, qd_pred_list = [], []
    yaw_errs, pos_errs = [], []

    with torch.no_grad():
        for k in range(args.steps):
            cmd = command_sampler(
                args.pattern,
                env.vehicle,
                rng,
                k,
                args.dt,
                args.period,
                args.fwd,
                args.diff,
                args.omega,
                args.thrust_commands,
            )

            sample = env.step(cmd)
            q_true, qdot_true, _ = env.to_planar(sample)

            # Model rollout
            R_flat = yaw_to_R_flat(q_pred[2:3]).to(device)
            state = torch.cat(
                (
                    q_pred[:2],
                    R_flat.squeeze(),
                    qd_pred,
                    torch.tensor(cmd, dtype=torch.float32, device=device),
                ),
                dim=0,
            )
            deriv = model(None, state.unsqueeze(0)).squeeze(0)
            state_next = state + args.dt * deriv

            x_next = state_next[0:2]
            R_next = state_next[2:6].view(2, 2)
            psi_next = torch.atan2(R_next[1, 0], R_next[0, 0])
            qd_next = state_next[6:9]

            q_pred = torch.stack((x_next[0], x_next[1], psi_next))
            qd_pred = qd_next

            qs_true.append(q_true)
            qs_pred.append(q_pred.cpu().numpy())
            qd_true_list.append(qdot_true)
            qd_pred_list.append(qd_next.cpu().numpy())

            yaw_err = wrap_angle(torch.tensor(float(q_pred[2].cpu().numpy()) - q_true[2]))
            yaw_errs.append(float(yaw_err))
            pos_errs.append(float(np.linalg.norm(q_true[:2] - q_pred.cpu().numpy()[:2])))

    qs_true = np.array(qs_true)
    qs_pred = np.array(qs_pred)
    qd_true_arr = np.array(qd_true_list)
    qd_pred_arr = np.array(qd_pred_list)
    yaw_errs = np.array(yaw_errs)
    pos_errs = np.array(pos_errs)

    vel_errs = np.linalg.norm(qd_true_arr - qd_pred_arr, axis=1)

    def rmse(arr):
        return float(np.sqrt(np.mean(arr ** 2)))

    print(f"Free-rollout RMSE over {args.steps} steps:")
    print(f"  Position RMSE: {rmse(pos_errs):.3f} m")
    print(f"  Yaw RMSE: {rmse(yaw_errs):.3f} rad")
    print(f"  Velocity RMSE: {rmse(vel_errs):.3f} m/s")

    t = np.arange(args.steps) * args.dt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("XY trajectory")
    plt.plot(qs_true[:, 0], qs_true[:, 1], label="sim")
    plt.plot(qs_pred[:, 0], qs_pred[:, 1], label="learned")
    plt.legend()
    plt.axis("equal")

    plt.subplot(2, 2, 2)
    plt.title("Position error norm (m)")
    plt.plot(t, pos_errs)

    plt.subplot(2, 2, 3)
    plt.title("Yaw error (rad)")
    plt.plot(t, yaw_errs)

    plt.subplot(2, 2, 4)
    plt.title("Velocity error norm (m/s)")
    plt.plot(t, vel_errs)

    plt.tight_layout()
    out_path = Path(args.output)
    plt.savefig(out_path, dpi=150)
    print(f"Saved free-rollout plot to {out_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="asv_framework/data/otter_se2_hamnode.pt")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pattern", type=str, default="zigzag", choices=["random", "zigzag", "figure8", "circle"])
    p.add_argument("--period", type=int, default=150, help="zigzag period (steps)")
    p.add_argument("--fwd", type=float, default=0.7, help="forward thrust fraction of n_max")
    p.add_argument("--diff", type=float, default=0.4, help="thrust differential fraction of n_max")
    p.add_argument("--omega", type=float, default=0.2, help="figure8 frequency (rad/s)")
    p.add_argument("--thrust-commands", action="store_true", help="interpret pattern outputs as thrust [N] instead of revs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="asv_framework/data/otter_free_rollout.png")
    return p.parse_args()


if __name__ == "__main__":
    rollout(get_args())
