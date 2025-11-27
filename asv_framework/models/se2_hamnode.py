"""
Hamiltonian Neural ODE on SE(2) for surface vehicle dynamics.

This mirrors the SE(3) HamNODE structure from LieGroupHamDL but reduces to
planar pose (x, y, yaw), body twist (u, v, r), and two control inputs.
"""
from __future__ import annotations

import torch

from asv_framework.paths import add_vendor_paths

add_vendor_paths()

from se3hamneuralode import MLP, PSD, MatrixNet  # type: ignore


def hat2(w: torch.Tensor) -> torch.Tensor:
    """Return 2x2 skew matrix for scalar angular rates."""
    w_flat = w.view(-1)
    z = torch.zeros_like(w_flat)
    return torch.stack(
        (
            torch.stack((z, -w_flat), dim=1),
            torch.stack((w_flat, z), dim=1),
        ),
        dim=1,
    )


def cross2_vec_scalar(vec: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Return w x vec in 2D (k-axis cross product)."""
    w_flat = w.view(-1, 1)
    return w_flat * torch.stack((-vec[:, 1], vec[:, 0]), dim=1)


def cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2D cross product returning a scalar per batch."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


class SE2HamNODE(torch.nn.Module):
    def __init__(
        self,
        device=None,
        udim: int = 2,
        turnon_dissipation: bool = True,
        hidden_dim: int = 256,
        init_gain: float = 0.01,
    ):
        super().__init__()
        self.device = device
        self.udim = udim
        self.xdim = 2
        self.Rdim = 4
        self.linveldim = 2
        self.angveldim = 1
        self.posedim = self.xdim + self.Rdim
        self.twistdim = self.linveldim + self.angveldim
        self.turnon_dissipation = turnon_dissipation

        # Mass Matrix (Inverse) - Constant in Body Frame
        # We use PSD with a dummy input of dimension 1 to learn a constant matrix
        self.M_net1 = PSD(
            1, hidden_dim, self.linveldim, init_gain=init_gain, epsilon=0.01
        ).to(device)
        self.M_net2 = PSD(
            1, hidden_dim, self.angveldim, init_gain=init_gain, epsilon=0.01
        ).to(device)

        if self.turnon_dissipation:
            # Damping - Depends on Body Velocity (nu)
            # Dv depends on linear velocity (2 dim)
            self.Dv_net = PSD(
                self.linveldim, hidden_dim, self.linveldim, init_gain=init_gain, epsilon=0.0
            ).to(device)
            # Dw depends on angular velocity (1 dim)
            self.Dw_net = PSD(
                self.angveldim, hidden_dim, self.angveldim, init_gain=init_gain, epsilon=0.0
            ).to(device)

        # Potential Energy - Assumed 0 for surface vessel
        # self.V_net = MLP(self.posedim, hidden_dim, 1, init_gain=init_gain).to(device)

        # Actuation Matrix - Constant in Body Frame
        # Input 1 -> Output Matrix
        self.g_net = MatrixNet(
            1,
            hidden_dim,
            self.twistdim * self.udim,
            shape=(self.twistdim, self.udim),
            init_gain=init_gain,
        ).to(device)
        self.nfe = 0

    def forward(self, t, input):
        with torch.enable_grad():
            self.nfe += 1
            q, q_dot, u = torch.split(
                input, [self.posedim, self.twistdim, self.udim], dim=1
            )
            x, R_flat = torch.split(q, [self.xdim, self.Rdim], dim=1)
            q_dot_v, q_dot_w = torch.split(
                q_dot, [self.linveldim, self.angveldim], dim=1
            )

            batch_size = q.shape[0]
            # Dummy input for constant networks
            dummy_input = torch.ones(batch_size, 1, device=self.device)

            # Mass Matrix (Inverse)
            M_q_inv1 = self.M_net1(dummy_input)
            M_q_inv2 = self.M_net2(dummy_input)
            M_q_inv2_scalar = M_q_inv2.view(batch_size, 1)

            # Compute Momentum p = M * nu => nu = M^-1 * p
            # Here we have nu (q_dot), so p = M * nu = (M^-1)^-1 * nu
            # BUT the code previously did pv = M_q_inv1 * q_dot_v.
            # This implies p = M^-1 * nu. This is dimensionally weird but let's stick to the
            # formulation that M_net outputs the METRIC tensor G^-1.
            # If H = 0.5 * p^T * M_net * p, then M_net is M^-1.
            # And p = dL/dv = M * v.
            # So v = M^-1 * p = M_net * p.
            # The previous code calculated pv from q_dot_v using M_net1.
            # pv = M_net1 * q_dot_v => p = M^-1 * v.
            # This means p is NOT momentum, or M_net1 is M (not M^-1).
            # Let's assume M_net1 is M (Mass Matrix).
            # Then H = 0.5 * p^T * M^-1 * p.
            # If M_net1 is M, then we need M^-1 for H.
            # Calculating inverse of network output is expensive/unstable.
            
            # ALTERNATIVE:
            # Let's assume M_net1 outputs M^-1 (Inverse Mass).
            # Then we need p = M * v = (M^-1)^-1 * v.
            # Inverting M_net1 output is better than inverting M.
            # Let's compute p by solving M^-1 * p = v.
            # pv = torch.linalg.solve(M_q_inv1, q_dot_v.unsqueeze(2)).squeeze(2)
            # pw = q_dot_w / M_q_inv2_scalar
            
            # HOWEVER, the previous code did: pv = M_q_inv1 * q_dot_v.
            # This implies p = M^-1 * v.
            # If we use this, then H = 0.5 * (M^-1 v)^T * M^-1 * (M^-1 v) = 0.5 v^T M^-1 v.
            # Kinetic energy is 0.5 v^T M v.
            # So this implies M^-1 = M. This is only true if M = I.
            # This suggests the previous code was fundamentally broken or I am misinterpreting.
            
            # Let's fix it properly.
            # M_net1 outputs M^-1 (Inverse Mass).
            # We want p = M * v.
            # So p = (M_net1)^-1 * v.
            
            # For 2x2 matrix, inverse is easy.
            # But let's stick to the structure:
            # Let's make M_net1 output M (Mass Matrix) directly?
            # No, H requires M^-1.
            
            # Let's make M_net1 output M^-1.
            # Then p = solve(M_net1, v).
            
            pv = torch.linalg.solve(M_q_inv1, torch.unsqueeze(q_dot_v, dim=2)).squeeze(2)
            pw = q_dot_w / M_q_inv2_scalar
            
            q_p = torch.cat((q, pv, pw), dim=1)
            x, R_flat, pv, pw = torch.split(
                q_p, [self.xdim, self.Rdim, self.linveldim, self.angveldim], dim=1
            )

            # Re-evaluate networks (they are constant/velocity dependent now)
            # M_q_inv1, M_q_inv2 are already computed.
            
            # Potential V(q) = 0
            V_q = torch.zeros(batch_size, 1, device=self.device)
            
            # Actuation
            g_q = self.g_net(dummy_input)
            
            if self.turnon_dissipation:
                # Damping depends on VELOCITY (q_dot_v, q_dot_w)
                Dv_net = self.Dv_net(q_dot_v)
                Dw_net = self.Dw_net(q_dot_w)

            p_aug_v = torch.unsqueeze(pv, dim=2)
            
            # Hamiltonian H = 0.5 * p^T * M^-1 * p
            H = (
                torch.squeeze(
                    torch.matmul(
                        torch.transpose(p_aug_v, 1, 2), torch.matmul(M_q_inv1, p_aug_v)
                    )
                )
                / 2.0
                + 0.5 * torch.squeeze(M_q_inv2_scalar) * torch.squeeze(pw) ** 2
                + torch.squeeze(V_q)
            )

            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True, allow_unused=True)[0]
            dHdx, dHdR, dHdpv, dHdpw = torch.split(
                dH, [self.xdim, self.Rdim, self.linveldim, self.angveldim], dim=1
            )
            
            # Handle unused gradients (e.g. dH/dq might be None if H doesn't depend on q)
            if dHdx is None: dHdx = torch.zeros(batch_size, self.xdim, device=self.device)
            if dHdR is None: dHdR = torch.zeros(batch_size, self.Rdim, device=self.device)
            if dHdpv is None: dHdpv = torch.zeros(batch_size, self.linveldim, device=self.device)
            if dHdpw is None: dHdpw = torch.zeros(batch_size, self.angveldim, device=self.device)

            Rmat = R_flat.view(-1, 2, 2)
            dx = torch.squeeze(torch.matmul(Rmat, torch.unsqueeze(dHdpv, dim=2)), dim=2)

            hat_w = hat2(dHdpw)
            dR = torch.matmul(Rmat, hat_w)
            dR = dR.reshape(batch_size, self.Rdim)

            F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)), dim=2)

            dpv = cross2_vec_scalar(pv, dHdpw) - torch.squeeze(
                torch.matmul(torch.transpose(Rmat, 1, 2), torch.unsqueeze(dHdx, dim=2)),
                dim=2,
            )
            if self.turnon_dissipation:
                # Damping force: - D(v) * v
                # Wait, usually it's -D(v) * v in the force equation.
                # The code has: - Dv_net * dHdpv.
                # dHdpv is velocity v (since H = 0.5 p^T M^-1 p => dH/dp = M^-1 p = v).
                # So this is correct: - D(v) * v.
                dpv = dpv - torch.squeeze(
                    torch.matmul(Dv_net, torch.unsqueeze(dHdpv, dim=2)), dim=2
                )
            dpv = dpv + F[:, 0:2]

            row0 = Rmat[:, 0, :]
            row1 = Rmat[:, 1, :]
            grad0 = dHdR[:, 0:2]
            grad1 = dHdR[:, 2:4]

            dpw = cross2(pv, dHdpv) + cross2(row0, grad0) + cross2(row1, grad1)
            if self.turnon_dissipation:
                dpw = dpw - torch.squeeze(
                    torch.matmul(Dw_net, torch.unsqueeze(dHdpw, dim=2)), dim=2
                )
            dpw = dpw.reshape(batch_size, -1).sum(dim=1, keepdim=True)
            dpw = dpw + F[:, 2].view(batch_size, 1)

            # d(M^-1)/dt terms
            # Since M^-1 is constant, dM_inv_dt is ZERO.
            # This simplifies things a lot.
            
            # dv = M^-1 * dpv + d(M^-1)/dt * pv
            dv = torch.squeeze(
                torch.matmul(M_q_inv1, torch.unsqueeze(dpv, dim=2)), dim=2
            )
            
            # dw = M^-1 * dpw + ...
            dw = M_q_inv2_scalar * dpw
            dw = dw.view(batch_size, 1)

            zero_vec = torch.zeros(
                batch_size, self.udim, dtype=torch.float32, device=self.device
            )
            batch_dims = (dx.shape[0], dR.shape[0], dv.shape[0], dw.shape[0], zero_vec.shape[0])
            if len(set(batch_dims)) != 1:
                raise RuntimeError(f"Batch size mismatch dx/dR/dv/dw/u: {batch_dims}")
            return torch.cat((dx, dR, dv, dw, zero_vec), dim=1)
