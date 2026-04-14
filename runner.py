"""Experiment runner: generate graphs, benchmark algorithms, collect results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from algorithms import (
    AStar,
    BFS,
    BellmanFord,
    BidirectionalDijkstra,
    Dijkstra,
    FloydWarshall,
)
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

    OPTIMAL_ALGOS = {
        "Dijkstra",
        "Bellman-Ford",
        "A*",
        "Bidirectional Dijkstra",
        "Floyd-Warshall",
    }

    def __init__(
        self,
        graphs_per_topology: int = 10,
        seed: int = 42,
        images_dir: str | Path = "graph_images",
    ):
        self.graphs_per_topology = graphs_per_topology
        self.generator = TopologyGenerator(seed=seed)
        self.images_dir = Path(images_dir)
        self.algorithms = [
            Dijkstra(),
            BellmanFord(),
            AStar(),
            BFS(),
            BidirectionalDijkstra(),
            FloydWarshall(),
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
                    image_path = self.generator.save_graph_image(
                        G,
                        source,
                        target,
                        topo_name,
                        graph_id,
                        self.images_dir / topo_name,
                    )
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
                                "graph_image": str(image_path),
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

    def export_analytics(self, df: pd.DataFrame, output_dir: str | Path) -> None:
        """Persist the key comparison tables for later analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        overall = self._aggregate_by(df, ["algorithm"]).sort_values("time_ms")
        by_topology = self._aggregate_by(df, ["topology", "algorithm"])
        by_algorithm = self._aggregate_by(df, ["algorithm", "topology"])

        overall.to_csv(output_dir / "overall_algorithm_efficiency.csv")
        by_topology.to_csv(output_dir / "algorithm_efficiency_per_topology.csv")
        by_algorithm.to_csv(output_dir / "algorithm_efficiency_between_topologies.csv")

        df.pivot_table(
            index="topology",
            columns="algorithm",
            values="execution_time_ms",
            aggfunc="mean",
        ).round(3).to_csv(output_dir / "execution_time_pivot.csv")

        df.pivot_table(
            index="topology",
            columns="algorithm",
            values="nodes_visited",
            aggfunc="mean",
        ).round(1).to_csv(output_dir / "nodes_visited_pivot.csv")

        df.pivot_table(
            index="topology",
            columns="algorithm",
            values="edges_relaxed",
            aggfunc="mean",
        ).round(1).to_csv(output_dir / "edges_relaxed_pivot.csv")

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return (
            df.groupby(keys)
            .agg(
                time_ms=("execution_time_ms", "mean"),
                nodes_visited=("nodes_visited", "mean"),
                edges_relaxed=("edges_relaxed", "mean"),
                path_cost=("path_cost", "mean"),
                path_hops=("path_hops", "mean"),
                success_rate=("success", "mean"),
            )
            .round(3)
        )

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

        # --- overall algorithm efficiency ---
        print("\n--- Overall algorithm efficiency (all topologies combined) ---\n")
        overall = ExperimentRunner._aggregate_by(df, ["algorithm"])
        print(overall.sort_values("time_ms").to_string())

        # --- compare algorithms within each topology ---
        print("\n--- Algorithm efficiency for each topology ---\n")
        topo_algo = ExperimentRunner._aggregate_by(df, ["topology", "algorithm"])
        print(topo_algo.to_string())

        # --- compare topologies within each algorithm ---
        print("\n--- Topology efficiency profile for each algorithm ---\n")
        algo_topo = ExperimentRunner._aggregate_by(df, ["algorithm", "topology"])
        print(algo_topo.to_string())

        print("\n--- Mean execution time (ms) : topology x algorithm ---\n")
        pivot_time = df.pivot_table(
            index="topology",
            columns="algorithm",
            values="execution_time_ms",
            aggfunc="mean",
        ).round(3)
        print(pivot_time.to_string())

        print("\n--- Mean nodes visited : topology x algorithm ---\n")
        pivot_nv = df.pivot_table(
            index="topology",
            columns="algorithm",
            values="nodes_visited",
            aggfunc="mean",
        ).round(1)
        print(pivot_nv.to_string())

        print("\n--- Mean edges relaxed : topology x algorithm ---\n")
        pivot_er = df.pivot_table(
            index="topology",
            columns="algorithm",
            values="edges_relaxed",
            aggfunc="mean",
        ).round(1)
        print(pivot_er.to_string())

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
