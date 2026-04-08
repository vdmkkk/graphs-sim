"""Experiment runner: generate graphs, benchmark algorithms, collect results."""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from algorithms import AStar, BFS, BellmanFord, BidirectionalDijkstra, Dijkstra
from graphs.generator import TopologyGenerator


class ExperimentRunner:
    """Orchestrates the full benchmark pipeline.

    Parameters
    ----------
    graphs_per_topology : int
        How many random graphs to create for each topology type.
    seed : int
        Master seed forwarded to :class:`TopologyGenerator`.
    """

    OPTIMAL_ALGOS = {"Dijkstra", "Bellman-Ford", "A*", "Bidirectional Dijkstra"}

    def __init__(self, graphs_per_topology: int = 100, seed: int = 42):
        self.graphs_per_topology = graphs_per_topology
        self.generator = TopologyGenerator(seed=seed)
        self.algorithms = [
            Dijkstra(),
            BellmanFord(),
            AStar(),
            BFS(),
            BidirectionalDijkstra(),
        ]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        topologies = self.generator.get_all_topologies()
        total = len(topologies) * self.graphs_per_topology
        records: list[dict] = []

        with tqdm(total=total, desc="Experiments", unit="graph") as pbar:
            for topo_name, gen_func in topologies:
                for graph_id in range(self.graphs_per_topology):
                    G, source, target = gen_func()
                    for algo in self.algorithms:
                        result = algo.run(G, source, target)
                        records.append(
                            {
                                "topology": topo_name,
                                "graph_id": graph_id,
                                "num_nodes": G.number_of_nodes(),
                                "num_edges": G.number_of_edges(),
                                "source": source,
                                "target": target,
                                "algorithm": result.algorithm,
                                "path_cost": result.path_cost,
                                "path_hops": result.path_hops,
                                "nodes_visited": result.nodes_visited,
                                "edges_relaxed": result.edges_relaxed,
                                "execution_time_ms": result.execution_time_ms,
                                "success": result.success,
                            }
                        )
                    pbar.update(1)

        df = pd.DataFrame(records)
        self._validate(df)
        return df

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        """Verify that all optimal algorithms agree on path cost."""
        optimal = df[df["algorithm"].isin(self.OPTIMAL_ALGOS)]
        grouped = optimal.groupby(["topology", "graph_id"])
        mismatches = 0
        for (topo, gid), grp in grouped:
            costs = grp[grp["success"]]["path_cost"].unique()
            if len(costs) > 1:
                mismatches += 1
                tqdm.write(
                    f"  WARNING: cost mismatch on {topo} graph#{gid}: {costs}"
                )
        if mismatches == 0:
            tqdm.write(
                "Validation OK - all optimal algorithms agree on shortest-path costs."
            )
        else:
            tqdm.write(f"Validation: {mismatches} cost mismatch(es) detected.")

    # ------------------------------------------------------------------
    # summary printing
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(df: pd.DataFrame) -> None:
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        n_topos = df["topology"].nunique()
        n_algos = df["algorithm"].nunique()
        gpg = df.groupby("topology")["graph_id"].nunique().iloc[0]
        print(f"\nGraphs per topology : {gpg}")
        print(f"Topologies          : {n_topos}")
        print(f"Algorithms          : {n_algos}")
        print(f"Total rows          : {len(df)}")

        # --- per-algorithm averages ---
        print("\n--- Mean metrics per algorithm (across all topologies) ---\n")
        agg = (
            df.groupby("algorithm")
            .agg(
                time_ms=("execution_time_ms", "mean"),
                nodes_visited=("nodes_visited", "mean"),
                edges_relaxed=("edges_relaxed", "mean"),
                path_cost=("path_cost", "mean"),
                path_hops=("path_hops", "mean"),
            )
            .round(3)
        )
        print(agg.to_string())

        # --- execution time pivot ---
        print("\n--- Mean execution time (ms) : topology x algorithm ---\n")
        pivot_time = df.pivot_table(
            index="topology",
            columns="algorithm",
            values="execution_time_ms",
            aggfunc="mean",
        ).round(3)
        print(pivot_time.to_string())

        # --- nodes-visited pivot ---
        print("\n--- Mean nodes visited : topology x algorithm ---\n")
        pivot_nv = df.pivot_table(
            index="topology",
            columns="algorithm",
            values="nodes_visited",
            aggfunc="mean",
        ).round(1)
        print(pivot_nv.to_string())

        # --- BFS sub-optimality ---
        bfs = (
            df[df["algorithm"] == "BFS (unweighted)"][
                ["topology", "graph_id", "path_cost"]
            ]
            .rename(columns={"path_cost": "bfs_cost"})
        )
        dij = (
            df[df["algorithm"] == "Dijkstra"][
                ["topology", "graph_id", "path_cost"]
            ]
            .rename(columns={"path_cost": "optimal_cost"})
        )
        merged = bfs.merge(dij, on=["topology", "graph_id"])
        merged["cost_ratio"] = merged["bfs_cost"] / merged["optimal_cost"]

        print("\n--- BFS path cost vs optimal (Dijkstra) ---\n")
        ratio = (
            merged.groupby("topology")["cost_ratio"]
            .agg(["mean", "median", "max"])
            .round(3)
        )
        ratio.columns = ["mean_ratio", "median_ratio", "max_ratio"]
        print(ratio.to_string())
