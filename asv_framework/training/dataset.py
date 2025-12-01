"""
Dataset utilities for Otter rollouts saved via collect_otter_rollouts.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class OtterStepDataset(Dataset):
    """
    Transitions from the rollout file.
    If horizon == 1: one-step pairs (q,q_dot,u) -> (q_next,q_dot_next).
    If horizon > 1: returns sequences of length horizon+1 for q/q_dot and horizon for u.
    """

    def __init__(self, npz_path: str, horizon: int = 1):
        data = np.load(npz_path)
        self.q = data["q"]  # shape: [episodes, steps, 3]
        self.q_dot = data["q_dot"]  # shape: [episodes, steps, 3]
        # Prefer actual actuator values if present; fall back to commanded for older files
        if "u_actual" in data:
            self.u = data["u_actual"]
        else:
            self.u = data["u"]
            
        if "u_control" in data:
            self.u_cmd = data["u_control"]
        elif "u_cmd" in data:
            self.u_cmd = data["u_cmd"]
        else:
            # Fallback if no command is stored (e.g. old data), assume u_cmd approx u_actual
            self.u_cmd = self.u

        # dt may be saved as a scalar or length-E array; take the first entry
        self.dt = float(np.asarray(data["dt"]).reshape(-1)[0])
        self.horizon = max(1, int(horizon))

        episodes, steps, _ = self.q.shape
        samples = []
        max_start = steps - self.horizon
        for ep in range(episodes):
            for k in range(max_start):
                samples.append((ep, k))
        self.index = samples

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, k = self.index[idx]
        if self.horizon == 1:
            q = torch.tensor(self.q[ep, k], dtype=torch.float32)
            q_dot = torch.tensor(self.q_dot[ep, k], dtype=torch.float32)
            u = torch.tensor(self.u[ep, k], dtype=torch.float32)
            u_cmd = torch.tensor(self.u_cmd[ep, k], dtype=torch.float32)

            q_next = torch.tensor(self.q[ep, k + 1], dtype=torch.float32)
            q_dot_next = torch.tensor(self.q_dot[ep, k + 1], dtype=torch.float32)

            y0 = torch.cat((q, q_dot, u), dim=0)
            y_target = torch.cat((q_next, q_dot_next), dim=0)
            # Return u_cmd as extra info
            return y0, y_target, self.dt, u_cmd
        else:
            q_seq = torch.tensor(self.q[ep, k : k + self.horizon + 1], dtype=torch.float32)
            qd_seq = torch.tensor(self.q_dot[ep, k : k + self.horizon + 1], dtype=torch.float32)
            # u_seq contains ACTUAL u for the sequence
            u_seq = torch.tensor(self.u[ep, k : k + self.horizon], dtype=torch.float32)
            # u_cmd_seq contains COMMANDED u for the sequence
            u_cmd_seq = torch.tensor(self.u_cmd[ep, k : k + self.horizon], dtype=torch.float32)
            return q_seq, qd_seq, u_seq, self.dt, u_cmd_seq
