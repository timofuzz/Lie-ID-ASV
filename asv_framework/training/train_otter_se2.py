"""
Training script for SE(2) Hamiltonian Neural ODE on Otter ASV data.

Supports:
- One-step prediction (horizon=1)
- Multi-step rollout training (horizon>1) with actuator dynamics
- Physics-informed diagnostics
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torch import amp

from asv_framework.models.se2_hamnode import SE2HamNODE
from asv_framework.training.dataset import OtterStepDataset

# Enable faster kernels when available
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def yaw_to_R_flat(psi: torch.Tensor) -> torch.Tensor:
    """Convert yaw angle to flattened 2x2 rotation matrix."""
    c = torch.cos(psi)
    s = torch.sin(psi)
    return torch.stack((c, -s, s, c), dim=1)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def integrate_one_step(model, state, dt: float, max_deriv: float):
    """
    Integrate one step using torchdiffeq (fixed-step RK4).
    Keeps the clamp on derivatives for stability.
    """
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


def prior_loss(model: SE2HamNODE, args, device):
    """Regularize masses and actuation toward physical priors to prevent collapse."""
    loss = 0.0

    if args.mass_prior_weight > 0 and model.use_learnable_mass:
        target_lin = torch.tensor(args.mass_prior_linear, device=device, dtype=torch.float32)
        target_ang = torch.tensor([args.mass_prior_angular], device=device, dtype=torch.float32)
        mass_lin = 1.0 / F.softplus(model.M_inv_linear)
        mass_ang = 1.0 / F.softplus(model.M_inv_angular)
        loss_m = F.mse_loss(mass_lin, target_lin) + F.mse_loss(mass_ang, target_ang)
        loss = loss + args.mass_prior_weight * loss_m

    if args.act_prior_weight > 0:
        dummy = torch.ones(1, 1, device=device)
        g = model.g_net(dummy)[0]
        target = torch.tensor(
            [
                [args.act_prior_surge, args.act_prior_surge],
                [0.0, 0.0],
                [args.act_prior_yaw, -args.act_prior_yaw],
            ],
            device=device,
            dtype=torch.float32,
        )
        loss_g = F.mse_loss(g, target)
        loss = loss + args.act_prior_weight * loss_g

    return loss


def diagnose_physics(model, device, epoch):
    """Print learned physical parameters."""
    params = model.get_learned_parameters()
    
    print(f"\nEpoch {epoch:03d} | Learned Physics Parameters")
    print(f"  Mass (linear): [{params['mass_linear'][0]:.1f}, {params['mass_linear'][1]:.1f}] kg "
          f"(expect: 50-70)")
    print(f"  Mass (angular): {params['mass_angular']:.2f} kg·m² (expect: 2-5)")
    print(f"  Actuation (surge): L={params['surge_left']:.4f}, R={params['surge_right']:.4f} "
          f"(expect: 0.02-0.05)")
    print(f"  Actuation (yaw):   L={params['yaw_left']:.4f}, R={params['yaw_right']:.4f} "
          f"(expect: ±0.01-0.03)")
    print(f"  Sign check: surge positive? {params['surge_left'] > 0 and params['surge_right'] > 0}, "
          f"yaw opposite? {params['yaw_left'] * params['yaw_right'] < 0}\n")


def train_one_step(model, loader, opt, scaler, use_amp, args):
    """Training loop for one-step prediction."""
    running_loss = 0.0
    
    for batch in loader:
        y0, y_target, dt, _ = batch
        y0 = y0.to(args.device)
        y_target = y_target.to(args.device)
        dt_tensor = dt.to(args.device).float().view(-1, 1)
        dt_scalar = float(dt_tensor[0].item())

        # Extract and normalize state
        q = y0[:, 0:3]
        q_dot = y0[:, 3:6]
        u = y0[:, 6:8]
        
        R_flat = yaw_to_R_flat(q[:, 2])
        state = torch.cat((
            q[:, 0:2] / args.pos_scale,
            R_flat,
            q_dot / args.vel_scale,
            u / args.u_scale
        ), dim=1)

        # Integrate one step with RK4
        with amp.autocast("cuda", enabled=use_amp):
            state_pred = integrate_one_step(model, state, dt_scalar, args.max_deriv)

        # Extract predictions
        x_pred = state_pred[:, 0:2] * args.pos_scale
        R_flat_pred = state_pred[:, 2:6]
        psi_pred = torch.atan2(R_flat_pred[:, 2], R_flat_pred[:, 0])
        q_dot_pred = state_pred[:, 6:9] * args.vel_scale

        # Extract targets
        x_true = y_target[:, 0:2]
        psi_true = y_target[:, 2]
        q_dot_true = y_target[:, 3:6]

        # Compute losses
        pos_loss = F.mse_loss(x_pred, x_true)
        yaw_loss = 35.0 * F.mse_loss(wrap_angle(psi_pred - psi_true), torch.zeros_like(psi_true))
        vel_loss = F.mse_loss(q_dot_pred, q_dot_true)
        vel_reg = args.vel_l2 * torch.mean(q_dot_pred ** 2)

        loss = pos_loss + yaw_loss + vel_loss + vel_reg
        loss = loss + prior_loss(model, args, args.device)

        if not torch.isfinite(loss):
            continue

        opt.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def train_multi_step(model, loader, opt, scaler, use_amp, args):
    """Training loop for multi-step rollout."""
    running_loss = 0.0
    T_n = 0.1  # Propeller time constant (from Otter specs)
    
    for batch in loader:
        q_seq, qd_seq, u_seq, dt, u_cmd_seq = batch
        q_seq = q_seq.to(args.device)
        qd_seq = qd_seq.to(args.device)
        u_seq = u_seq.to(args.device)
        u_cmd_seq = u_cmd_seq.to(args.device)
        dt_tensor = dt.to(args.device).float().view(-1, 1)
        dt_scalar = float(dt_tensor[0].item())

        # Initialize state
        q_pred = q_seq[:, 0, :]
        qd_pred = qd_seq[:, 0, :]
        u_pred = u_seq[:, 0, :]

        loss_terms = []
        
        for h in range(args.horizon):
            # Actuator dynamics: ṅ = (n_cmd - n) / T_n
            n_dot = (u_cmd_seq[:, h, :] - u_pred) / T_n
            u_next = u_pred + dt_tensor * n_dot
            
            # Build normalized state
            R_flat = yaw_to_R_flat(q_pred[:, 2])
            state = torch.cat((
                q_pred[:, 0:2] / args.pos_scale,
                R_flat,
                qd_pred / args.vel_scale,
                u_pred / args.u_scale
            ), dim=1)
            
            # Integrate one step with RK4
            with amp.autocast("cuda", enabled=use_amp):
                state_next = integrate_one_step(model, state, dt_scalar, args.max_deriv)
            if not torch.isfinite(state_next).all():
                break

            # Extract predictions
            x_next = state_next[:, 0:2] * args.pos_scale
            R_next = state_next[:, 2:6].view(-1, 2, 2)
            psi_next = torch.atan2(R_next[:, 1, 0], R_next[:, 0, 0])
            
            # Re-orthogonalize rotation
            R_proj = yaw_to_R_flat(psi_next)
            qd_next = state_next[:, 6:9] * args.vel_scale

            # Targets
            x_true = q_seq[:, h + 1, 0:2]
            psi_true = q_seq[:, h + 1, 2]
            qd_true = qd_seq[:, h + 1, :]

            # Losses
            pos_loss = F.mse_loss(x_next, x_true)
            yaw_loss = 100.0 * F.mse_loss(wrap_angle(psi_next - psi_true), torch.zeros_like(psi_true))
            vel_loss = F.mse_loss(qd_next, qd_true)
            vel_reg = args.vel_l2 * torch.mean(qd_next ** 2)
            
            loss_terms.append(pos_loss + yaw_loss + vel_loss + vel_reg)

            # Update state for next step
            q_pred = torch.stack((x_next[:, 0], x_next[:, 1], psi_next), dim=1)
            qd_pred = qd_next
            u_pred = u_next

        if len(loss_terms) == 0:
            continue
            
        loss = torch.stack(loss_terms).mean()
        loss = loss + prior_loss(model, args, args.device)

        if not torch.isfinite(loss):
            continue

        opt.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train SE(2) HamNODE for Otter ASV")
    
    # Data
    parser.add_argument("--data", type=str, required=True, help="path to npz file")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--horizon", type=int, default=10, help="rollout length")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="number of initial epochs to train with horizon=1 before switching to --horizon")
    
    # Model
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--use-learnable-mass",
        dest="use_learnable_mass",
        action="store_true",
        default=True,
        help="use simple parameters instead of network for mass (default: True)",
    )
    parser.add_argument(
        "--no-learnable-mass",
        dest="use_learnable_mass",
        action="store_false",
        help="use the neural mass network with prior instead of simple parameters",
    )
    
    # Regularization
    parser.add_argument("--max-deriv", type=float, default=200.0)
    parser.add_argument("--vel-l2", type=float, default=0.0)
    parser.add_argument("--mass-prior-linear", nargs=2, type=float, default=[50.0, 50.0],
                        help="target surge/sway mass (kg) for prior loss")
    parser.add_argument("--mass-prior-angular", type=float, default=3.0,
                        help="target yaw inertia (kg·m²) for prior loss")
    parser.add_argument("--mass-prior-weight", type=float, default=1e-3,
                        help="weight of mass prior loss (0 to disable)")
    parser.add_argument("--act-prior-surge", type=float, default=0.03,
                        help="target surge gain for each propeller")
    parser.add_argument("--act-prior-yaw", type=float, default=0.001,
                        help="target yaw gain magnitude per propeller (signs enforced by structure)")
    parser.add_argument("--act-prior-weight", type=float, default=1e-3,
                        help="weight of actuation prior loss (0 to disable)")
    
    # Scaling
    parser.add_argument("--pos-scale", type=float, default=150.0)
    parser.add_argument("--vel-scale", type=float, default=2.0)
    parser.add_argument("--u-scale", type=float, default=50.0)
    
    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, required=True)
    
    args = parser.parse_args()

    # Dataset inspection
    data = np.load(args.data)
    print(f"\n{'='*50}")
    print(f"Dataset Statistics")
    print(f"{'='*50}")
    print(f"Episodes: {data['q'].shape[0]}, Steps: {data['q'].shape[1]}")
    print(f"Position range: [{data['q'][:,:,:2].min():.1f}, {data['q'][:,:,:2].max():.1f}] m")
    print(f"Velocity range: [{data['q_dot'].min():.2f}, {data['q_dot'].max():.2f}] m/s")
    
    u_key = 'u_actual' if 'u_actual' in data else 'u'
    print(f"Actuator ({u_key}): [{data[u_key].min():.1f}, {data[u_key].max():.1f}] rad/s")
    print(f"{'='*50}\n")

    # Build model and optimizer
    dataset_main = OtterStepDataset(args.data, horizon=args.horizon)
    loader_main = DataLoader(
        dataset_main,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    loader_warmup = None
    if args.warmup_epochs > 0 and args.horizon > 1:
        dataset_warmup = OtterStepDataset(args.data, horizon=1)
        loader_warmup = DataLoader(
            dataset_warmup,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    model = SE2HamNODE(
        device=args.device,
        hidden_dim=args.hidden_dim,
        use_learnable_mass=args.use_learnable_mass,
        pos_scale=args.pos_scale,
        vel_scale=args.vel_scale,
        u_scale=args.u_scale,
    ).to(args.device)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    print(f"Training with horizon={args.horizon}, lr={args.lr}, batch_size={args.batch_size}\n")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        use_warmup = args.horizon > 1 and args.warmup_epochs > 0 and epoch <= args.warmup_epochs
        if use_warmup:
            avg_loss = train_one_step(model, loader_warmup, opt, scaler, use_amp, args)
            current_h = 1
        elif args.horizon == 1:
            avg_loss = train_one_step(model, loader_main, opt, scaler, use_amp, args)
            current_h = 1
        else:
            avg_loss = train_multi_step(model, loader_main, opt, scaler, use_amp, args)
            current_h = args.horizon

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            diagnose_physics(model, args.device, epoch)
            print(f"  Loss (h={current_h}): {avg_loss:.6f}\n")
        else:
            print(f"Epoch {epoch:03d} [h={current_h}] | Loss: {avg_loss:.6f}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.save_path)

    # Final save
    torch.save(model.state_dict(), args.save_path)
    print(f"\n{'='*50}")
    print(f"Training complete! Model saved to {args.save_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
