"""Routing Algorithm Simulator — entry point.

Compare shortest-path algorithms across diverse graph topologies.

Usage
-----
    python main.py                         # 100 graphs/topology, seed 42
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
        default=100,
        help="graphs to generate per topology (default: 100)",
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
    args = parser.parse_args()

    print(f"Graphs/topology : {args.graphs_per_topology}")
    print(f"Seed            : {args.seed}")
    print(f"Output          : {args.output}")
    print()

    runner = ExperimentRunner(
        graphs_per_topology=args.graphs_per_topology,
        seed=args.seed,
    )

    df = runner.run()
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    runner.print_summary(df)


if __name__ == "__main__":
    main()
