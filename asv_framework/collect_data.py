"""
Collect training data from the Otter simulator.
"""
import argparse
import numpy as np
from pathlib import Path
from asv_framework.otter_env import OtterEnv
from asv_framework.controls import command_sampler

def collect(args):
    env = OtterEnv(sample_time=args.dt)
    rng = np.random.default_rng(args.seed)
    
    all_q = []
    all_q_dot = []
    all_u_actual = []
    all_u_cmd = []
    all_dt = []
    
    patterns = ["random", "zigzag", "figure8", "circle"]
    
    print(f"Collecting {args.episodes} episodes...")
    
    for ep in range(args.episodes):
        env.reset()
        # Randomize initial state slightly
        env.eta[5] = rng.uniform(-np.pi, np.pi)
        
        # Pick a pattern
        if args.pattern:
            pattern = args.pattern
        else:
            pattern = rng.choice(patterns)
        
        # Use command-line parameters or randomize
        if args.randomize_params:
            period = rng.integers(100, 300)
            fwd = rng.uniform(0.4, 0.9)
            diff = rng.uniform(0.2, 0.5)
            omega = rng.uniform(0.1, 0.4)
        else:
            period = args.period
            fwd = args.fwd
            diff = args.diff
            omega = args.omega
        
        qs, qds, u_acts, u_cmds = [], [], [], []
        
        for k in range(args.steps):
            cmd = command_sampler(
                pattern,
                env.vehicle,
                rng,
                k,
                args.dt,
                period,
                fwd,
                diff,
                omega,
            )
            
            sample = env.step(cmd)
            q, q_dot, u_cmd, u_actual = env.to_planar(sample)
            
            qs.append(q)
            qds.append(q_dot)
            u_acts.append(u_actual)
            u_cmds.append(u_cmd)
            
        all_q.append(qs)
        all_q_dot.append(qds)
        all_u_actual.append(u_acts)
        all_u_cmd.append(u_cmds)
        all_dt.append(args.dt)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{args.episodes} done.")

    # Save to npz
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        out_path,
        q=np.array(all_q),
        q_dot=np.array(all_q_dot),
        u_actual=np.array(all_u_actual),
        u_control=np.array(all_u_cmd),
        dt=np.array(all_dt)
    )
    print(f"Saved data to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect training data from Otter simulator")
    
    # Basic parameters
    p.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect")
    p.add_argument("--steps", type=int, default=2000, help="Steps per episode")
    p.add_argument("--dt", type=float, default=0.05, help="Simulation timestep (seconds)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--output", type=str, default="asv_framework/data/otter_rollouts.npz", 
                   help="Output file path")
    
    # Pattern selection
    p.add_argument("--pattern", type=str, default=None, 
                   choices=["random", "zigzag", "figure8", "circle"], 
                   help="Force a specific pattern (if None, randomly samples)")
    
    # Pattern parameters
    p.add_argument("--period", type=int, default=150, 
                   help="Zigzag/circle period in steps (default: 150)")
    p.add_argument("--fwd", type=float, default=0.7, 
                   help="Forward thrust fraction [0-1] (default: 0.7)")
    p.add_argument("--diff", type=float, default=0.4, 
                   help="Differential thrust fraction [0-1] (default: 0.4)")
    p.add_argument("--omega", type=float, default=0.3, 
                   help="Angular frequency for figure-8 pattern (default: 0.3)")
    
    # Randomization control
    p.add_argument("--randomize-params", action="store_true", 
                   help="Randomize period/fwd/diff/omega for each episode (overrides explicit values)")
    
    args = p.parse_args()
    collect(args)
