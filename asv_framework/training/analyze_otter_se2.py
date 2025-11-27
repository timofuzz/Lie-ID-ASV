import torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from asv_framework.training.dataset import OtterStepDataset
from asv_framework.training.train_otter_se2 import yaw_to_R_flat, wrap_angle
from asv_framework.models.se2_hamnode import SE2HamNODE

DATA = "asv_framework/data/otter_rollouts.npz"
CKPT = "asv_framework/data/otter_se2_hamnode.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 200  # steps to roll out for comparison
SAVE_FIG = Path("asv_framework/data/otter_analysis.png")

ds = OtterStepDataset(DATA)
data = np.load(DATA)
q = data["q"][0]      # episode 0
q_dot = data["q_dot"][0]
u = data["u"][0]
dt = float(np.asarray(data["dt"]).reshape(-1)[0])

model = SE2HamNODE(device=DEVICE).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

max_horizon = min(HORIZON, len(u) - 1)

# One-step prediction with teacher forcing (uses true state each step)
q_pred_list, qdot_pred_list = [], []
with torch.no_grad():
    for k in range(max_horizon):
        q_curr = torch.tensor(q[k], dtype=torch.float32, device=DEVICE)
        qd_curr = torch.tensor(q_dot[k], dtype=torch.float32, device=DEVICE)
        R_flat = yaw_to_R_flat(q_curr[2:3]).to(DEVICE)
        state = torch.cat(
            (
                q_curr[:2],
                R_flat.squeeze(),
                qd_curr,
                torch.tensor(u[k], dtype=torch.float32, device=DEVICE),
            ),
            dim=0,
        )
        deriv = model(None, state.unsqueeze(0)).squeeze(0)
        state_next = state + dt * deriv

        x_next = state_next[0:2]
        R_next = state_next[2:6].view(2, 2)
        psi_next = torch.atan2(R_next[1, 0], R_next[0, 0])
        qd_next = state_next[6:9]

        q_pred = torch.stack((x_next[0], x_next[1], psi_next))
        q_pred_list.append(q_pred.cpu().numpy())
        qdot_pred_list.append(qd_next.cpu().numpy())

q_true = q[1:max_horizon+1]
psi_true = q_true[:,2]
q_pred = np.array(q_pred_list)
psi_pred = q_pred[:,2]
# unwrap yaw error
yaw_err = wrap_angle(torch.tensor(psi_pred - psi_true)).numpy()

# Error metrics (ignore NaNs/infs from unstable rollout)
pos_err = np.linalg.norm(q_true[:,0:2] - q_pred[:,0:2], axis=1)
vel_true = q_dot[1:max_horizon+1]
vel_pred = np.array(qdot_pred_list)
vel_err = np.linalg.norm(vel_true - vel_pred, axis=1)

def finite_rmse(arr):
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(arr[mask]**2)))

print(f"Position RMSE: {finite_rmse(pos_err):.3f} m")
print(f"Yaw RMSE: {finite_rmse(yaw_err):.3f} rad")
print(f"Velocity RMSE: {finite_rmse(vel_err):.3f} m/s")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("XY trajectory")
plt.plot(q_true[:, 0], q_true[:, 1], label="sim")
plt.plot(q_pred[:, 0], q_pred[:, 1], label="learned")
plt.legend()
plt.axis("equal")
# Focus limits around observed data
x_all = np.concatenate([q_true[:,0], q_pred[:,0]])
y_all = np.concatenate([q_true[:,1], q_pred[:,1]])
if np.all(np.isfinite(x_all)) and np.all(np.isfinite(y_all)):
    pad = 0.5
    plt.xlim(x_all.min()-pad, x_all.max()+pad)
    plt.ylim(y_all.min()-pad, y_all.max()+pad)
plt.subplot(1, 2, 2)
plt.title("Yaw error (rad)")
plt.plot(yaw_err)
plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=150)
print(f"Saved plot to {SAVE_FIG}")
