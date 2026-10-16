"""Model 2 entry point — LoRa mesh protocol-suitability simulator.

Usage
-----
    python -m model2_protocols.main                       # full study
    python -m model2_protocols.main --quick               # fast smoke run
    python -m model2_protocols.main --sf 12 --nodes 60    # custom PHY/size
    python -m model2_protocols.main --no-sweep            # skip the load sweep
"""

from __future__ import annotations

import argparse

from .config import SimConfig, LoRaPHY
from .network.deployment import reliable_range_m
from .runner import ProtocolExperiment


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate routing protocols / network stacks for LoRa mesh.")
    p.add_argument("--sf", type=int, default=9, choices=range(7, 13),
                   help="LoRa spreading factor (default: 9)")
    p.add_argument("--nodes", type=int, default=25, help="nodes per scenario")
    p.add_argument("--time", type=float, default=1200.0,
                   help="simulated seconds per run (default: 1200)")
    p.add_argument("--reps", type=int, default=10,
                   help="repetitions per cell (for confidence intervals)")
    p.add_argument("--seed", type=int, default=42, help="master seed")
    p.add_argument("--out", type=str, default="model2_protocols/results",
                   help="output directory")
    p.add_argument("--no-sweep", action="store_true", help="skip the load sweep")
    p.add_argument("--no-extra", action="store_true",
                   help="skip deployment / node / SF sweeps")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke configuration (small + short)")
    args = p.parse_args()

    if args.quick:
        args.nodes, args.time, args.reps = 25, 400.0, 2
        args.no_extra = True

    cfg = SimConfig(phy=LoRaPHY(sf=args.sf), seed=args.seed, sim_time_s=args.time)
    exp = ProtocolExperiment(cfg=cfg, reps=args.reps, num_nodes=args.nodes)

    print(f"SF={args.sf}  nodes={args.nodes}  sim_time={args.time}s  reps={args.reps}")
    print(f"reliable range ~ {reliable_range_m(cfg):.0f} m")
    print()

    import pandas as pd
    matrix = exp.run_matrix()
    sweep = exp.run_load_sweep() if not args.no_sweep else pd.DataFrame()

    deployment = node_sweep = sf_sweep = None
    if not args.no_extra:
        deployment = exp.run_deployment_sweep()
        node_sweep = exp.run_node_sweep()
        sf_sweep = exp.run_sf_sweep()

    exp.export_analytics(matrix, sweep, args.out, deployment=deployment,
                         node_sweep=node_sweep, sf_sweep=sf_sweep)
    exp.print_summary(matrix)
    print(f"\nArtefacts written to {args.out}/")


if __name__ == "__main__":
    main()
