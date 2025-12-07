from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from asv_framework.paths import add_vendor_paths
add_vendor_paths()


def choose_nonlinearity(name):
    """Select activation function by name."""
    activations = {
        'tanh': torch.tanh,
        'relu': torch.relu,
        'sigmoid': torch.sigmoid,
        'softplus': F.softplus,
        'selu': F.selu,
        'elu': F.elu,
        'swish': lambda x: x * torch.sigmoid(x)
    }
    if name not in activations:
        raise ValueError(f"Unknown nonlinearity: {name}")
    return activations[name]


class MLP(nn.Module):
    """Three-layer multilayer perceptron with configurable activation."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 nonlinearity='tanh', bias_bool=True, init_gain=1.0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=bias_bool)

        for layer in [self.linear1, self.linear2, self.linear3]:
            nn.init.orthogonal_(layer.weight, gain=init_gain)

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)


class PSD(nn.Module):
    """
    Positive semi-definite matrix network: M = LL^T + εI
    
    Ensures learned matrices (mass, damping) are always positive-definite.
    """
    
    def __init__(self, input_dim, hidden_dim, diag_dim, 
                 nonlinearity='tanh', init_gain=0.1, epsilon=0.1):
        super().__init__()
        self.diag_dim = diag_dim
        self.epsilon = epsilon
        
        if diag_dim == 1:
            # Scalar case (angular inertia)
            self.net = MLP(input_dim, hidden_dim, 1, 
                          nonlinearity, init_gain=init_gain)
        else:
            # Matrix case (linear mass/damping)
            self.off_diag_dim = diag_dim * (diag_dim - 1) // 2
            output_dim = diag_dim + self.off_diag_dim
            self.net = MLP(input_dim, hidden_dim, output_dim,
                          nonlinearity, init_gain=init_gain)

    def forward(self, x):
        """Compute PSD matrix from input."""
        batch_size = x.shape[0]
        
        if self.diag_dim == 1:
            # Scalar output - RETURN AS (batch, 1, 1) for consistency
            raw = self.net(x)
            scalar_value = F.softplus(raw) + self.epsilon
            return scalar_value.view(batch_size, 1, 1)  # ← FIX: Add view()
        else:
            # Matrix output: construct lower-triangular L, then LL^T
            raw = self.net(x)
            diag, off_diag = torch.split(raw, [self.diag_dim, self.off_diag_dim], dim=1)
            
            # Ensure positive diagonal
            diag = F.softplus(diag).to(dtype=x.dtype)
            off_diag = off_diag.to(dtype=x.dtype)
            
            # Build lower-triangular matrix
            L = torch.diag_embed(diag)
            ind = np.tril_indices(self.diag_dim, k=-1)
            flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            L_flat = L.flatten(start_dim=1).to(dtype=x.dtype)
            L_flat[:, flat_ind] = off_diag
            L = L_flat.view(batch_size, self.diag_dim, self.diag_dim)
            
            # Compute LL^T + εI
            M = torch.bmm(L, L.transpose(1, 2))
            eye = torch.eye(self.diag_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
            return M + self.epsilon * eye


class StructuredActuationNet(nn.Module):
    """
    Learns actuation matrix g(q) with differential-drive structure.
    
    Physical constraints:
    - Both propellers produce forward thrust (surge > 0)
    - No lateral actuation (sway ≈ 0)
    - Yaw control via opposite propeller moments
    """
    
    def __init__(self, hidden_dim, init_gain=0.1):
        super().__init__()
        # Learn surge coefficients [k_left, k_right]
        self.surge_net = MLP(1, hidden_dim, 2, init_gain=init_gain)
        
        # Learn yaw moment arms [d_left, d_right]
        self.yaw_net = MLP(1, hidden_dim, 2, init_gain=init_gain)

        # Scaling to keep outputs near physical magnitudes
        self.surge_scale = 0.05
        self.yaw_scale = 0.02
        
        # Initialize near physical values
        with torch.no_grad():
            self.surge_net.linear3.bias.data = torch.tensor([0.03, 0.03])
            self.yaw_net.linear3.bias.data = torch.tensor([0.015, -0.015])
    
    def forward(self, dummy):
        """
        Construct 3x2 actuation matrix:
        g = [[surge_L,  surge_R ],
             [   0,        0     ],
             [ yaw_L,   yaw_R   ]]
        """
        batch_size = dummy.shape[0]
        
        # Surge: force positive, allow asymmetry
        surge = self.surge_scale * F.softplus(self.surge_net(dummy))  # (batch, 2)
        
        # Sway: forced to zero (no lateral thrust)
        sway = torch.zeros_like(surge)
        
        # Yaw: enforce opposite signs but learn magnitudes independently
        yaw_raw = self.yaw_net(dummy)
        yaw_left = self.yaw_scale * F.softplus(yaw_raw[:, 0:1])
        yaw_right = -self.yaw_scale * F.softplus(yaw_raw[:, 1:2])
        yaw = torch.cat([yaw_left, yaw_right], dim=1)
        
        # Stack into 3x2 matrix
        return torch.stack([surge, sway, yaw], dim=1)


class SE2HamNODE(nn.Module):
    """
    Port-Hamiltonian Neural ODE on SE(2) for Otter ASV.
    
    State: [x, y, R11, R12, R21, R22, u, v, r, n_left, n_right]
           └── pose (6D) ──┘ └─ velocity (3D) ─┘ └─ control (2D) ─┘
    
    Learns:
    - M(q): Mass matrix (with prior for rigid body)
    - D(v): Velocity-dependent damping
    - g(q): Structured actuation matrix
    
    Physics:
    - Hamiltonian structure: M v̇ = F_control + F_damping + F_coriolis
    - Coriolis forces include mass: C(v) = ad_v^*(M)
    - Energy conservation (when D=0)
    """
    
    def __init__(self, device=None, udim=2, turnon_dissipation=True,
                 hidden_dim=256, init_gain=0.01, use_learnable_mass=True,
                 pos_scale: float = 1.0, vel_scale: float = 1.0, u_scale: float = 1.0):
        super().__init__()
        self.device = device
        self.udim = udim
        self.xdim = 2          # Position (x, y)
        self.Rdim = 4          # Rotation matrix flattened
        self.linveldim = 2     # Linear velocity (u, v)
        self.angveldim = 1     # Angular velocity (r)
        self.posedim = self.xdim + self.Rdim
        self.twistdim = self.linveldim + self.angveldim
        self.turnon_dissipation = turnon_dissipation
        self.use_learnable_mass = use_learnable_mass
        self.pos_scale = float(pos_scale)
        self.vel_scale = float(vel_scale)
        self.u_scale = float(u_scale)
        
        # ===================================================================
        # MASS MATRIX: Two approaches (toggle via use_learnable_mass)
        # ===================================================================
        if use_learnable_mass:
            # Simple learnable parameters (fast, reliable)
            init_lin_inv = torch.log(torch.expm1(torch.tensor([0.015, 0.015])))
            init_ang_inv = torch.log(torch.expm1(torch.tensor([0.30])))
            self.M_inv_linear = nn.Parameter(init_lin_inv)
            self.M_inv_angular = nn.Parameter(init_ang_inv)
            print("[INFO] Using learnable mass parameters (3 DOF)")
        else:
            # Neural network with physics prior (more general)
            init_base = torch.log(torch.expm1(torch.tensor([0.015, 0.015, 0.30])))
            self.M_base = nn.Parameter(init_base)
            self.M_correction = MLP(1, 128, 3, init_gain=0.01)
            print("[INFO] Using neural mass network with prior")
        
        # ===================================================================
        # DAMPING MATRIX (velocity-dependent)
        # ===================================================================
        if turnon_dissipation:
            self.Dv_net = PSD(self.linveldim, hidden_dim, self.linveldim,
                            init_gain=init_gain, epsilon=0.1)
            self.Dw_net = PSD(self.angveldim, hidden_dim, self.angveldim,
                            init_gain=init_gain, epsilon=0.1)
        
        # ===================================================================
        # ACTUATION MATRIX (structured for differential drive)
        # ===================================================================
        self.g_net = StructuredActuationNet(hidden_dim, init_gain=0.1)
    
    def get_mass_matrix_inv(self, batch_size, dummy):
        """Compute inverse mass matrix M^{-1}."""
        if self.use_learnable_mass:
            # Simple parameter approach
            M_inv1 = torch.diag_embed(
                F.softplus(self.M_inv_linear).expand(batch_size, -1)
            )
            M_inv2 = F.softplus(self.M_inv_angular).view(1, 1, 1).expand(batch_size, -1, -1)
        else:
            # Network with prior approach
            M_inv_base = self.M_base.expand(batch_size, -1)
            M_inv_correction = 0.1 * self.M_correction(dummy)
            M_inv_combined = F.softplus(M_inv_base + M_inv_correction)
            M_inv1 = torch.diag_embed(M_inv_combined[:, :2])
            M_inv2 = M_inv_combined[:, 2:3].unsqueeze(2)
        
        return M_inv1, M_inv2
    
    def forward(self, t, state):
        """
        Compute time derivative: ẏ = f(y, t)
        
        Args:
            t: time (unused, for ODE solver)
            state: [batch, 11] = [x, y, R_flat(4), u, v, r, n_L, n_R]
        
        Returns:
            deriv: [batch, 11] time derivatives
        """
        batch_size = state.shape[0]
        dummy = torch.ones(batch_size, 1, device=state.device)
        
        # ===============================================================
        # EXTRACT STATE
        # ===============================================================
        x_norm = state[:, 0:2]               # Normalized position
        R_flat = state[:, 2:6]               # Rotation matrix
        v_lin_norm = state[:, 6:8]           # Normalized linear velocity
        v_ang_norm = state[:, 8:9]           # Normalized angular velocity
        u_norm = state[:, 9:11]              # Normalized control inputs
        
        R = R_flat.view(-1, 2, 2)
        v_lin = v_lin_norm * self.vel_scale
        v_ang = v_ang_norm * self.vel_scale
        u = u_norm * self.u_scale
        
        # ===============================================================
        # MASS MATRIX
        # ===============================================================
        M_inv1, M_inv2 = self.get_mass_matrix_inv(batch_size, dummy)
        
        # Extract mass values for Coriolis (M = (M_inv)^{-1})
        M_diag = 1.0 / torch.diagonal(M_inv1, dim1=-2, dim2=-1)
        m_surge = M_diag[:, 0:1]
        m_sway = M_diag[:, 1:2]
        I_inv = M_inv2.view(batch_size, 1)
        
        # ===============================================================
        # ACTUATION FORCES
        # ===============================================================
        # Thrust model: F_prop ~ sign(n) * n^2
        u_squared = torch.sign(u) * (u ** 2)
        g_q = self.g_net(dummy)  # (batch, 3, 2)
        F = torch.bmm(g_q, u_squared.unsqueeze(2)).squeeze(2)
        
        F_lin = F[:, 0:2]
        F_ang = F[:, 2:3]
        
        # ===============================================================
        # DAMPING FORCES
        # ===============================================================
        if self.turnon_dissipation:
            D_v = self.Dv_net(v_lin)  # (batch, 2, 2)
            damping_lin = -torch.bmm(D_v, v_lin.unsqueeze(2)).squeeze(2)
            
            # FIX: Handle scalar angular damping
            D_w_raw = self.Dw_net(v_ang)  # (batch, 1, 1) or (batch, 1)
            if D_w_raw.dim() == 2:
                # Scalar case: just multiply
                damping_ang = -D_w_raw * v_ang
            else:
                # Matrix case: use bmm
                damping_ang = -torch.bmm(D_w_raw, v_ang.unsqueeze(2)).squeeze(2)
        else:
            damping_lin = torch.zeros_like(F_lin)
            damping_ang = torch.zeros_like(F_ang)
        
        # ===============================================================
        # CORIOLIS FORCES (with mass terms)
        # ===============================================================
        u_vel = v_lin[:, 0:1]
        v_vel = v_lin[:, 1:2]
        r_vel = v_ang[:, 0:1]
        
        coriolis_lin = torch.cat([
            -m_sway * r_vel * v_vel,
             m_surge * r_vel * u_vel
        ], dim=1)
        coriolis_ang = torch.zeros_like(v_ang)
        
        # ===============================================================
        # ACCELERATIONS (Hamiltonian equations)
        # ===============================================================
        total_force_lin = F_lin + damping_lin + coriolis_lin
        a_lin = torch.bmm(M_inv1, total_force_lin.unsqueeze(2)).squeeze(2)
        
        total_force_ang = F_ang + damping_ang + coriolis_ang
        a_ang = I_inv * total_force_ang
        
        # ===============================================================
        # POSE DERIVATIVES (Lie group kinematics)
        # ===============================================================
        # Position: ẋ = R @ v_body
        x_dot = torch.bmm(R, v_lin.unsqueeze(2)).squeeze(2)
        
        # Rotation: Ṙ = R @ [r]_×
        r = v_ang[:, 0]
        skew = torch.stack([
            torch.stack([torch.zeros_like(r), -r], dim=1),
            torch.stack([r, torch.zeros_like(r)], dim=1)
        ], dim=1)
        R_dot = torch.bmm(R, skew).view(-1, 4)
        
        # ===============================================================
        # ASSEMBLE OUTPUT
        # ===============================================================
        q_dot = torch.cat([x_dot / self.pos_scale, R_dot], dim=1)
        v_dot = torch.cat([a_lin, a_ang], dim=1) / self.vel_scale
        u_dot = torch.zeros_like(u_norm)  # Control is external (normalized space)
        
        return torch.cat([q_dot, v_dot, u_dot], dim=1)
    
    def get_learned_parameters(self):
        """Extract learned physics parameters for inspection."""
        with torch.no_grad():  # Disable gradient tracking
            dummy = torch.ones(1, 1, device=self.device)
            
            # Mass matrix
            M_inv1, M_inv2 = self.get_mass_matrix_inv(1, dummy)
            M_linear = torch.linalg.inv(M_inv1.squeeze()).diag().cpu().numpy()
            M_angular = (1.0 / M_inv2.squeeze()).cpu().item()
            
            # Actuation matrix
            g = self.g_net(dummy).squeeze().cpu().numpy()
            
            return {
                'mass_linear': M_linear,
                'mass_angular': M_angular,
                'actuation_matrix': g,
                'surge_left': g[0, 0],
                'surge_right': g[0, 1],
                'yaw_left': g[2, 0],
                'yaw_right': g[2, 1],
            }
