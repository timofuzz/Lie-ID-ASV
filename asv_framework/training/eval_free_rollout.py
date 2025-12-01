"""
Free-rollout evaluation of the learned SE(2) HamNODE against the Otter simulator.

Compares multi-step trajectories under scripted excitation patterns (random, zigzag, figure8)
and reports RMSE over the horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from asv_framework.otter_env import OtterEnv
from asv_framework.training.train_otter_se2 import yaw_to_R_flat, wrap_angle
from asv_framework.models.se2_hamnode import SE2HamNODE
from asv_framework.controls import command_sampler


def integrate_one_step(model, state, dt: float, max_deriv: float):
    """One-step RK4 integration with clamped derivatives."""
    t_span = torch.tensor([0.0, dt], device=state.device, dtype=state.dtype)

    def ode_func(t, y):
        deriv = model(t, y)
        return torch.clamp(deriv, -max_deriv, max_deriv)

    sol = odeint(
        ode_func,
        state,
        t_span,
        method="rk4",
        options={"step_size": dt},
    )
    return sol[-1]


def rollout(args):
    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    env = OtterEnv(sample_time=args.dt, use_autopilot=False)
    env.reset()
    rng = np.random.default_rng(args.seed)

    model = SE2HamNODE(
        device=device,
        hidden_dim=args.hidden_dim,
        pos_scale=args.pos_scale,
        vel_scale=args.vel_scale,
        u_scale=args.u_scale,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Get initial state from simulator
    initial_sample = {"eta": env.eta, "nu": env.nu, "u_control": np.zeros(2), "u_actual": np.zeros(2)}
    q_init, qd_init, _, _ = env.to_planar(initial_sample)
    
    # Initialize predicted state from simulator's initial state
    q_pred = torch.tensor(q_init, dtype=torch.float32, device=device)
    qd_pred = torch.tensor(qd_init, dtype=torch.float32, device=device)

    qs_true, qs_pred = [], []
    qd_true_list, qd_pred_list = [], []
    u_cmd_list, u_actual_list = [], []
    yaw_errs, pos_errs = [], []

    # Initialize predicted actuator state
    u_pred_val = np.array(initial_sample["u_actual"], dtype=float)

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
            )

            # Step simulator for ground truth
            sample = env.step(cmd)
            q_true, qdot_true, u_cmd_sim, u_actual_sim = env.to_planar(sample)

            # --- Explicit Actuator Dynamics (Open-Loop) ---
            # Propeller dynamics: n_dot = (n_cmd - n) / T_n
            # We use the same T_n as the simulator (0.1s)
            T_n = 0.1
            n_dot = (cmd - u_pred_val) / T_n
            u_pred_next_val = u_pred_val + args.dt * n_dot
            
            # Model rollout using PREDICTED actuator state
            R_flat = yaw_to_R_flat(q_pred[2:3]).to(device)
            u_tensor = torch.tensor(u_pred_val, dtype=torch.float32, device=device)
            u_proc = u_tensor / args.u_scale
            state = torch.cat(
                (
                    q_pred[:2] / args.pos_scale,
                    R_flat.squeeze(),
                    qd_pred / args.vel_scale,
                    u_proc,
                ),
                dim=0,
            )
            state_next = integrate_one_step(
                model,
                state.unsqueeze(0),
                args.dt,
                args.max_deriv,
            ).squeeze(0)

            x_next = state_next[0:2] * args.pos_scale
            R_next = state_next[2:6].view(2, 2)
            psi_next = torch.atan2(R_next[1, 0], R_next[0, 0])
            # Re-project rotation from yaw to avoid skew accumulation
            R_next = yaw_to_R_flat(psi_next.unsqueeze(0)).view(2, 2)
            qd_next = state_next[6:9] * args.vel_scale

            q_pred = torch.stack((x_next[0], x_next[1], psi_next))
            qd_pred = qd_next
            
            # Update actuator state for next step
            u_pred_val = u_pred_next_val

            qs_true.append(q_true)
            qs_pred.append(q_pred.cpu().numpy())
            qd_true_list.append(qdot_true)
            qd_pred_list.append(qd_next.cpu().numpy())
            u_cmd_list.append(cmd)
            u_actual_list.append(u_actual_sim) # Keep plotting sim actual for comparison

            yaw_err = wrap_angle(torch.tensor(float(q_pred[2].cpu().numpy()) - q_true[2]))
            yaw_errs.append(float(yaw_err))
            pos_errs.append(float(np.linalg.norm(q_true[:2] - q_pred.cpu().numpy()[:2])))

    qs_true = np.array(qs_true)
    qs_pred = np.array(qs_pred)
    qd_true_arr = np.array(qd_true_list)
    qd_pred_arr = np.array(qd_pred_list)
    yaw_errs = np.array(yaw_errs)
    pos_errs = np.array(pos_errs)
    u_cmd_arr = np.array(u_cmd_list)
    u_actual_arr = np.array(u_actual_list)

    vel_errs = np.linalg.norm(qd_true_arr - qd_pred_arr, axis=1)

    def rmse(arr):
        return float(np.sqrt(np.mean(arr ** 2)))

    print(f"Free-rollout RMSE over {args.steps} steps:")
    print(f"  Position RMSE: {rmse(pos_errs):.3f} m")
    print(f"  Yaw RMSE: {rmse(yaw_errs):.3f} rad")
    print(f"  Velocity RMSE: {rmse(vel_errs):.3f} m/s")

    t = np.arange(args.steps) * args.dt
    plt.figure(figsize=(12, 9))
    plt.subplot(3, 2, 1)
    plt.title("XY trajectory")
    plt.plot(qs_true[:, 0], qs_true[:, 1], label="sim")
    plt.plot(qs_pred[:, 0], qs_pred[:, 1], label="learned")
    plt.legend()
    plt.axis("equal")

    plt.subplot(3, 2, 2)
    plt.title("Position error norm (m)")
    plt.plot(t, pos_errs)

    plt.subplot(3, 2, 3)
    plt.title("Yaw error (rad)")
    plt.plot(t, yaw_errs)

    plt.subplot(3, 2, 4)
    plt.title("Velocity error norm (m/s)")
    plt.plot(t, vel_errs)

    plt.subplot(3, 2, 5)
    plt.title("Prop 1 input (rad/s)")
    plt.plot(t, u_cmd_arr[:, 0], label="commanded")
    plt.plot(t, u_actual_arr[:, 0], label="actual")
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.title("Prop 2 input (rad/s)")
    plt.plot(t, u_cmd_arr[:, 1], label="commanded")
    plt.plot(t, u_actual_arr[:, 1], label="actual")
    plt.legend()

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
    p.add_argument("--hidden-dim", type=int, default=256, help="hidden dimension used during training")
    p.add_argument("--pattern", type=str, default="zigzag", choices=["random", "zigzag", "figure8", "circle"])
    p.add_argument("--period", type=int, default=150, help="zigzag period (steps)")
    p.add_argument("--fwd", type=float, default=0.7, help="forward rev fraction of n_max")
    p.add_argument("--diff", type=float, default=0.3, help="rev differential fraction of n_max")
    p.add_argument("--omega", type=float, default=0.2, help="figure8 frequency (rad/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="asv_framework/data/otter_free_rollout.png")
    p.add_argument("--pos-scale", type=float, default=150.0, help="divide x,y by this before feeding the model")
    p.add_argument("--vel-scale", type=float, default=2.0, help="divide u,v,r by this before feeding the model")
    p.add_argument("--u-scale", type=float, default=50.0, help="divide control inputs by this")
    p.add_argument("--max-deriv", type=float, default=200.0, help="clamp on predicted derivative during eval")
    return p.parse_args()


if __name__ == "__main__":
    rollout(get_args())
