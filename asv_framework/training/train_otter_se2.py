"""
One-step training loop for the SE(2) HamNODE using Otter rollouts.

This uses forward-Euler on the Neural ODE to predict the next state and
matches the pose/velocity targets from the simulator data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from asv_framework.models.se2_hamnode import SE2HamNODE
from asv_framework.training.dataset import OtterStepDataset


def yaw_to_R_flat(psi: torch.Tensor) -> torch.Tensor:
    c = torch.cos(psi)
    s = torch.sin(psi)
    return torch.stack((c, -s, s, c), dim=1)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "otter_rollouts.npz"),
        help="path to rollout npz file",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=1, help="short-horizon rollout length for training")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "data" / "otter_se2_hamnode.pt"
        ),
        help="where to store the trained model weights",
    )
    args = parser.parse_args()

    dataset = OtterStepDataset(args.data, horizon=args.horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = SE2HamNODE(
        device=args.device, udim=2, turnon_dissipation=True, hidden_dim=args.hidden_dim
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch in loader:
            if args.horizon == 1:
                y0, y_target, dt = batch
                y0 = y0.to(args.device)
                y_target = y_target.to(args.device)
                dt_tensor = dt.to(args.device).float().view(-1, 1)

                q = y0[:, 0:3]
                q_dot = y0[:, 3:6]
                u = y0[:, 6:8]
                R_flat = yaw_to_R_flat(q[:, 2])
                q_full = torch.cat((q[:, 0:2], R_flat), dim=1)
                state = torch.cat((q_full, q_dot, u), dim=1)

                deriv = model(None, state)
                state_pred = state + dt_tensor * deriv

                # Extract predictions
                x_pred = state_pred[:, 0:2]
                R_flat_pred = state_pred[:, 2:6]
                psi_pred = torch.atan2(R_flat_pred[:, 2], R_flat_pred[:, 0])
                q_dot_pred = state_pred[:, 6:9]

                x_true = y_target[:, 0:2]
                psi_true = y_target[:, 2]
                q_dot_true = y_target[:, 3:6]

                pos_loss = F.mse_loss(x_pred, x_true)
                yaw_loss = F.mse_loss(wrap_angle(psi_pred - psi_true), torch.zeros_like(psi_true))
                vel_loss = F.mse_loss(q_dot_pred, q_dot_true)

                loss = pos_loss + yaw_loss + vel_loss
            else:
                # short-horizon free rollout loss
                q_seq, qd_seq, u_seq, dt = batch
                q_seq = q_seq.to(args.device)         # shape: [B, H+1, 3]
                qd_seq = qd_seq.to(args.device)       # shape: [B, H+1, 3]
                u_seq = u_seq.to(args.device)         # shape: [B, H, 2]
                dt_tensor = dt.to(args.device).float().view(-1, 1)

                # teacher-forced multi-step: predict next state from true state at each step
                loss_terms = []
                for h in range(args.horizon):
                    q_curr = q_seq[:, h, :]
                    qd_curr = qd_seq[:, h, :]

                    R_flat = yaw_to_R_flat(q_curr[:, 2])
                    state = torch.cat((q_curr[:, 0:2], R_flat, qd_curr, u_seq[:, h, :]), dim=1)
                    deriv = model(None, state)

                    if not torch.isfinite(deriv).all():
                        continue

                    state_pred = state + dt_tensor * deriv

                    x_pred = state_pred[:, 0:2]
                    R_flat_pred = state_pred[:, 2:6]
                    psi_pred = torch.atan2(R_flat_pred[:, 2], R_flat_pred[:, 0])
                    q_dot_pred = state_pred[:, 6:9]

                    x_true = q_seq[:, h + 1, 0:2]
                    psi_true = q_seq[:, h + 1, 2]
                    q_dot_true = qd_seq[:, h + 1, :]

                    pos_loss = F.mse_loss(x_pred, x_true)
                    yaw_loss = F.mse_loss(wrap_angle(psi_pred - psi_true), torch.zeros_like(psi_true))
                    vel_loss = F.mse_loss(q_dot_pred, q_dot_true)
                    loss_terms.append(pos_loss + yaw_loss + vel_loss)

                if len(loss_terms) == 0:
                    continue
                loss = torch.stack(loss_terms).mean()

            if not torch.isfinite(loss):
                # Skip unstable batches
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch:03d} | loss {avg_loss:.6f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
