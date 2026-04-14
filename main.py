"""Routing Algorithm Simulator — entry point.

Compare shortest-path algorithms across diverse graph topologies.

Usage
-----
    python main.py                         # 10 graphs/topology, seed 42
    python main.py -n 200 -s 7 -o run.csv  # customise
"""

import argparse

from runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark shortest-path algorithms on random graph topologies.",
    )
    parser.add_argument(
        "-n",
        "--graphs-per-topology",
        type=int,
        default=10,
        help="graphs to generate per topology (default: 10)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="master random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results.csv",
        help="output CSV path (default: results.csv)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="graph_images",
        help="directory for generated graph images (default: graph_images)",
    )
    parser.add_argument(
        "--analytics-dir",
        type=str,
        default="analytics",
        help="directory for generated analytics tables (default: analytics)",
    )
    args = parser.parse_args()

    print(f"Graphs/topology : {args.graphs_per_topology}")
    print(f"Seed            : {args.seed}")
    print(f"Output          : {args.output}")
    print(f"Images dir      : {args.images_dir}")
    print(f"Analytics dir   : {args.analytics_dir}")
    print()

    runner = ExperimentRunner(
        graphs_per_topology=args.graphs_per_topology,
        seed=args.seed,
        images_dir=args.images_dir,
    )

    df = runner.run()
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    runner.export_analytics(df, args.analytics_dir)

    runner.print_summary(df)


if __name__ == "__main__":
    main()
