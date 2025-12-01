# Lie-ID-ASV: Port-Hamiltonian System ID for the Otter ASV

Learn a structured SE(2) Hamiltonian model of the Otter surface vehicle using rollouts from the `python_vehicle_simulator`.

## Overview
- Collect excitation data from the Otter simulator (`asv_framework/otter_env.py` + `asv_framework/controls.py` patterns).
- Train a structured SE(2) Hamiltonian Neural ODE with learned mass, damping, and actuation (`asv_framework/models/se2_hamnode.py`) using fixed-step RK4 (`torchdiffeq`).
- Evaluate open-loop free rollouts against the simulator (`asv_framework/training/eval_free_rollout.py`).

## Requirements
```
pip install numpy torch matplotlib torchdiffeq
```
Ensure the vendored simulator and HamDL repo are on `PYTHONPATH` (handled via `asv_framework/paths.py` when you import the package).

## Workflow
1) Collect data
```bash
python -m asv_framework.collect_data --episodes 50 --steps 2000 \
  --pattern zigzag --period 150 --fwd 0.7 --diff 0.4 \
  --output asv_framework/data/otter_zigzag.npz
```
2) Train
```bash
python -m asv_framework.training.train_otter_se2 --data asv_framework/data/otter_zigzag.npz  --epochs 30 --batch-size 512 --lr 5e-5 --hidden-dim 512 --horizon 10 --warmup-epochs 5  --pos-scale 150.0 --vel-scale 2.0 --u-scale 50.0 --max-deriv 150.0  --act-prior-weight 1e-2 --mass-prior-weight 1e-3 --device cuda --save-path asv_framework/data/otter_model.pt
```
3) Evaluate free rollout
```bash
python -m asv_framework.training.eval_free_rollout --model asv_framework/data/otter_model.pt --steps 400 --pattern zigzag --period 150 --fwd 0.7 --diff 0.3 --hidden-dim 512 --pos-scale 150.0 --vel-scale 2.0 --u-scale 50.0 --output asv_framework/data/eval_plot.png
```

## Key options (training)
- `--horizon`: rollout length during training (higher improves long-term accuracy; combine with `--warmup-epochs` for stability).
- `--pos-scale`, `--vel-scale`, `--u-scale`: normalize inputs; keep consistent between train and eval.
- `--max-deriv`: clamps model derivatives during integration.
- `--use-learnable-mass` (default) / `--no-learnable-mass`: choose simple scalar masses or the NN-with-prior variant.
- Priors: `--mass-prior-*`, `--act-prior-*` keep learned parameters near physical ranges.

## Model notes
- Actuation matrix enforces surge-only thrust and opposite yaw moments; thrust is applied as `n * |n|` internally.
- Damping is velocity-dependent PSD for linear and angular channels.
- Integrator: fixed-step RK4 via `torchdiffeq` (manual Euler removed) with derivative clamping; actuator lag is modeled during train/eval.

## Troubleshooting
- Divergence: increase `--horizon`, reduce `--lr`, or lower `--max-deriv`.
- Scale mismatch: ensure the same `pos/vel/u` scales are used for train and eval.
- Weak thrust estimates: raise `--act-prior-weight` slightly or reduce `u-scale` so inputs land near order-of-magnitude forces.

## File map
- `asv_framework/otter_env.py`: simulator wrapper.
- `asv_framework/controls.py`: shared thruster command sampler (random/zigzag/figure8/circle).
- `asv_framework/collect_data.py`: dataset generation.
- `asv_framework/training/train_otter_se2.py`: training loop.
- `asv_framework/training/eval_free_rollout.py`: open-loop evaluation.
- `asv_framework/models/se2_hamnode.py`: Hamiltonian model definition.
- `asv_framework/paths.py`: adds vendor repos to `sys.path`.
