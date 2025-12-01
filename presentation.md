# Slide 1 — Title
**Port-Hamiltonian System ID for the Otter ASV (SE(2))**  
Structured neural ODE with physical priors, trained on simulator rollouts.

# Slide 2 — Motivation
- Underactuated catamaran (2 props, no sway thrust) needs reliable ID for planning/control.
- Pure black-box models drift; structure-preserving dynamics are more stable and interpretable.
- Goal: learn mass, damping, actuation while respecting SE(2) geometry and thrust physics.

# Slide 3 — Simulator & Data
- Simulator: `python_vehicle_simulator` Otter (`otter_env.py` wrapper), prop lag T_n = 0.1 s.
- Inputs: prop rev commands → `u_actual`; Outputs: planar state q = [x, y, ψ], q̇ = [u, v, r].
- Command patterns: random / zigzag / figure8 / circle (shared sampler in `controls.py`).
- Dataset example: 100 episodes × 4000 steps, dt = 0.05 s, zigzag with strong differential.

# Slide 4 — State & Model Inputs
- State to network: normalized [x, y, R_flat(2×2), u, v, r, n1, n2].
- Normalization: pos/vel/u scales (e.g., pos 150, vel 2, u 40–50); must match at eval.
- Actuator state carried through rollout with lag T_n = 0.1 (matches sim).

# Slide 5 — Model Architecture
- SE(2) Hamiltonian Neural ODE (`se2_hamnode.py`):
  - Mass: positive diagonal (or NN variant) with softplus.
  - Damping: PSD nets for linear/angular drag.
  - Actuation: structured 3×2, surge-only thrust, opposite-sign yaw; thrust law n|n|.
  - Coriolis from mass; no sway actuation.
- Integration: fixed-step RK4 (torchdiffeq) with derivative clamp for stability.

# Slide 6 — Training Loop
- Multi-step rollout loss (horizon 10–30) after warmup (h=1).
- Loss terms: position MSE, wrapped yaw MSE, velocity MSE, optional vel L2.
- Physics priors: L2 toward target masses (≈60 kg, 3 kg·m²) and actuation gains.
- AMP + dataloader workers for speed; grad clipping to 1.0.

# Slide 7 — Key Hyperparameters
- horizon / warmup-epochs balance stability vs. long-horizon fit.
- pos-scale / vel-scale / u-scale must match train & eval.
- max-deriv clamp (e.g., 100–200) avoids explosive integration.
- act-prior-* and mass-prior-* set strength of physics nudges.
- yaw loss weight (can be increased if heading is critical).

# Slide 8 — What the Data Says (Yaw)
- Regression on dataset: ṙ ≈ 7.3e-05 · (n1|n1| − n2|n2|) − 2.39 · r.
- Implication: simulator yaw torque per rev² is tiny; “expected 0.01–0.03” was too high.
- Actuation priors for yaw should be ~1e-4–1e-3, or accept small yaw authority.

# Slide 9 — Results (Example Zigzag)
- Trained 50 epochs, horizon 30, u-scale 40–50, act_prior_weight 5e-2, mass_prior_weight 1e-3.
- Eval (matching scales) on zigzag 400 steps:
  - Position RMSE ≈ 2–3 m
  - Yaw RMSE ≈ 0.08–0.16 rad
  - Velocity RMSE ≈ 0.2–0.3 m/s
- Trajectory shape captured; drift remains due to low yaw authority in sim physics.

# Slide 10 — Ablations Tried
- Longer horizon (10 → 30): better long-horizon shape, slower training.
- Actuation priors up/down: surge learns near targets; yaw stays small (consistent with data).
- u-scale changes: helps conditioning but doesn’t change physical yaw torque in data.
- Different datasets (stronger differential): modest improvement, still low yaw gain.

# Slide 11 — Limitations
- Simulator’s yaw torque coefficient is low; learned yaw authority is bounded by data.
- Structured actuation forbids sway thrust; any unmodeled lateral effects stay unmodeled.
- RK4 fixed step; no learned integrator or adaptive stepping.
- No disturbance/current modeling; purely open-loop rollout evaluation.

# Slide 12 — How to Improve Further
- If higher yaw is desired: adjust thrust→torque mapping (coefficients) or increase yaw_scale in actuation; otherwise accept measured scale.
- Increase yaw loss weight modestly; ensure eval scales match training.
- Collect richer yaw excitation (higher diff, figure8) if staying within same physics.
- Consider learning a small residual on actuation or damping if matching real hardware.

# Slide 13 — Practical Recipe (Current Best)
- Collect: zigzag, dt=0.05, diff≈0.5, 100×4000 steps.
- Train: horizon 20–30, warmup 5–10, pos 150, vel 2, u 40–50, max-deriv 100–150,
  act-prior-yaw ~1e-3, act-prior-weight ~1e-2, mass-prior-weight ~1e-3, device=cuda.
- Eval: match all scales; use same T_n=0.1; patterns comparable to training.

# Slide 14 — Takeaways
- Structure (SE(2), PSD mass/damping, underactuated g) + priors yields stable ID.
- The simulator physics set an upper bound on yaw authority; match priors to measured torque.
- Long-horizon rollouts are reasonably accurate; residual drift reflects low yaw torque and open-loop evaluation.

# Slide 15 — References / Code Map
- Code: `asv_framework/models/se2_hamnode.py`, `training/train_otter_se2.py`, `training/eval_free_rollout.py`, `collect_data.py`, `controls.py`, `otter_env.py`.
- Simulator: `PythonVehicleSimulator` (`vehicles/otter.py`) with T_n = 0.1, k_pos/k_neg, lever arms.
