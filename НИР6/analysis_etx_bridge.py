# -*- coding: utf-8 -*-
"""Cross-model bridge (point 7): run Model 1's shortest-path algorithms on the
*physically-grounded ETX-weighted* graphs produced by Model 2.

This ties the two models into one story: Model 1 ranks the algorithms on
abstract random weights; here we check that the same ranking holds when the
weights are real "expected airtime per delivery" (ETX) values from the LoRa
radio model of Model 2.

Run from the repo root with the project venv:
    python analysis_etx_bridge.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model1_algorithms"))   # expose the `algorithms` pkg

from algorithms import (  # noqa: E402
    AStar, BFS, BellmanFord, BidirectionalDijkstra, Dijkstra, FloydWarshall,
)
from model2_protocols.config import SimConfig  # noqa: E402
from model2_protocols.network.deployment import (  # noqa: E402
    DeploymentScenario, build_network,
)

ALGORITHMS = [Dijkstra(), BellmanFord(), AStar(), BFS(),
              BidirectionalDijkstra(), FloydWarshall()]
OPTIMAL = {"Dijkstra", "Bellman-Ford", "A*", "Bidirectional Dijkstra",
           "Floyd-Warshall"}


def main() -> None:
    cfg = SimConfig()
    deployments = [DeploymentScenario.UNIFORM, DeploymentScenario.CORRIDOR,
                   DeploymentScenario.CLUSTERED]
    num_nodes = 40
    seeds = [42, 43, 44]
    pairs_per_graph = 12

    rows = []
    mismatches = 0
    for dep in deployments:
        for seed in seeds:
            net = build_network(cfg, num_nodes, dep, rng=random.Random(seed))
            G = net.ground_truth_graph(only_operational=False)  # ETX weights
            # candidate (s,t) pairs within the same connected component
            pairs = []
            for comp in nx.connected_components(G):
                comp = list(comp)
                for a in range(len(comp)):
                    for b in range(a + 1, len(comp)):
                        pairs.append((comp[a], comp[b]))
            if not pairs:
                continue
            rng = random.Random(seed * 100)
            chosen = rng.sample(pairs, min(pairs_per_graph, len(pairs)))
            for src, dst in chosen:
                costs = {}
                for algo in ALGORITHMS:
                    res = algo.run(G, src, dst)
                    rows.append({
                        "deployment": dep.value, "seed": seed,
                        "src": src, "dst": dst,
                        "algorithm": res.algorithm,
                        "path_cost": round(res.path_cost, 4) if res.success else None,
                        "path_hops": res.path_hops,
                        "nodes_visited": res.nodes_visited,
                        "edges_relaxed": res.edges_relaxed,
                        "execution_time_ms": res.execution_time_ms,
                        "success": res.success,
                    })
                    if res.success and res.algorithm in OPTIMAL:
                        costs[res.algorithm] = round(res.path_cost, 3)
                if len(set(costs.values())) > 1:
                    mismatches += 1

    df = pd.DataFrame(rows)
    out_dir = ROOT / "etx_bridge"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "etx_bridge_raw.csv", index=False)

    summary = (df.groupby("algorithm")
               .agg(time_ms=("execution_time_ms", "mean"),
                    nodes_visited=("nodes_visited", "mean"),
                    edges_relaxed=("edges_relaxed", "mean"),
                    path_cost=("path_cost", "mean"),
                    path_hops=("path_hops", "mean"),
                    success_rate=("success", "mean"))
               .round(3).sort_values("nodes_visited"))
    summary.to_csv(out_dir / "etx_bridge_summary.csv")

    print("=" * 70)
    print("ETX-BRIDGE: Model 1 algorithms on Model 2 physical (ETX) weights")
    print("=" * 70)
    print(f"graphs: {len(deployments)*len(seeds)}  pairs/graph<= {pairs_per_graph}"
          f"  total queries: {len(df)//len(ALGORITHMS)}")
    print(f"optimal-cost disagreements: {mismatches}")
    print()
    print(summary.to_string())

    # plot: nodes visited by algorithm on ETX weights
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        o = summary.reset_index().sort_values("nodes_visited")
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(o["algorithm"], o["nodes_visited"], color="#6a5acd")
        ax.set_ylabel("Среднее число посещённых вершин")
        ax.set_title("Алгоритмы Модели 1 на реальных ETX-весах Модели 2")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "etx_bridge_nodes.png", dpi=130)
        plt.close(fig)
        print(f"\nArtefacts written to {out_dir}/")
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
