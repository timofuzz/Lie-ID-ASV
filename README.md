## Lie-ID-ASV: Otter ASV system ID + control scaffold

This repo glues three upstream projects into a Python-only workflow:

- `PythonVehicleSimulator-master` – ground-truth Otter USV dynamics (Fossen)
- `LieGroupHamDL-main` – Hamiltonian Neural ODE framework (adapted to SE(2))
- `Lie-MPC-AMVs-main` – Matlab MPC reference (kept for comparison)

### Layout added here
- `asv_framework/otter_env.py` – Gym-like Otter wrapper for fast rollouts
- `asv_framework/data/collect_otter_rollouts.py` – collect excitation data to `.npz`
- `asv_framework/models/se2_hamnode.py` – SE(2) HamNODE (planar pose/yaw, u,v,r)
- `asv_framework/training/dataset.py` – dataset helper for rollouts
- `asv_framework/training/train_otter_se2.py` – one-step HamNODE trainer

### Quick start
1) Collect data (rich thrust excitation):
```bash
python -m asv_framework.data.collect_otter_rollouts --episodes 5 --steps 2000 --dt 0.05
```
This saves `asv_framework/data/otter_rollouts.npz`.

2) Train the SE(2) HamNODE:
```bash
python -m asv_framework.training.train_otter_se2 --data asv_framework/data/otter_rollouts.npz --epochs 30 --device cuda
```
Weights land at `asv_framework/data/otter_se2_hamnode.pt` (configurable).

Install deps: `pip install -r requirements.txt` (Torch build can be chosen per your CUDA setup).

### Notes
- `asv_framework.paths.add_vendor_paths()` injects `PythonVehicleSimulator-master/src` and `LieGroupHamDL-main` so imports work without moving code.
- The trainer uses forward-Euler against one-step targets; swap in `torchdiffeq.odeint` over short horizons if you want multi-step matching.
- Controls are appended to the state with zero derivative (like the quadrotor code), so keep integration windows aligned with the held input or extend the model to interpolate time-varying `u`.

### Next steps
- Add an NMPC/MPPI controller that rolls out either the ground-truth `otter_env` or the learned `SE2HamNODE`.
- Expand the dataset to cover current profiles/payload shifts; include those as conditioning inputs to the networks if you need a single robust model.
