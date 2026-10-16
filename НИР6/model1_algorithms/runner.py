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

        # Overall efficiency with dispersion (mean, std, n) so the report can
        # quote mean +/- spread instead of a bare point estimate.
        overall_stats = (
            df.groupby("algorithm")
            .agg(
                time_ms_mean=("execution_time_ms", "mean"),
                time_ms_std=("execution_time_ms", "std"),
                nodes_mean=("nodes_visited", "mean"),
                nodes_std=("nodes_visited", "std"),
                edges_mean=("edges_relaxed", "mean"),
                edges_std=("edges_relaxed", "std"),
                n=("execution_time_ms", "count"),
            )
            .round(3)
            .sort_values("time_ms_mean")
        )
        overall_stats.to_csv(output_dir / "overall_efficiency_with_ci.csv")

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------
    def export_plots(self, df: pd.DataFrame, plots_dir: str | Path) -> None:
        """Reproducible figures for the report: overall bars + per-topology heatmaps."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:  # pragma: no cover - plotting is optional
            tqdm.write("matplotlib/numpy unavailable - skipping Model 1 plots")
            return

        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        overall = (df.groupby("algorithm")
                   .agg(time=("execution_time_ms", "mean"),
                        time_std=("execution_time_ms", "std"),
                        nodes=("nodes_visited", "mean"),
                        nodes_std=("nodes_visited", "std"))
                   .reset_index())

        # 1) execution time (log scale) with std whiskers
        o = overall.sort_values("time")
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(o["algorithm"], o["time"], yerr=o["time_std"], capsize=3,
               color="#3b6fb5")
        ax.set_yscale("log")
        ax.set_ylabel("Среднее время выполнения, мс (лог. шкала)")
        ax.set_title("Среднее время поиска пути по алгоритмам")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout(); fig.savefig(plots_dir / "m1_time.png", dpi=130); plt.close(fig)

        # 2) nodes visited with std whiskers
        o = overall.sort_values("nodes")
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(o["algorithm"], o["nodes"], yerr=o["nodes_std"], capsize=3,
               color="#5aa469")
        ax.set_ylabel("Среднее число посещённых вершин")
        ax.set_title("Среднее число посещённых вершин (вычислительные затраты)")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout(); fig.savefig(plots_dir / "m1_nodes.png", dpi=130); plt.close(fig)

        # 3) heatmaps algorithm x topology (nodes visited, edges relaxed)
        def heatmap(value, fname, title, cbar, log):
            piv = df.pivot_table(index="topology", columns="algorithm",
                                 values=value, aggfunc="mean")
            piv = piv.sort_index()
            arr = piv.values.astype(float)
            disp = np.log10(np.clip(arr, 1e-9, None)) if log else arr
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            im = ax.imshow(disp, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels(piv.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(piv.index)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    txt = f"{v:.0f}" if v >= 10 else f"{v:.2g}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5)
            ax.set_title(title)
            cb = fig.colorbar(im, ax=ax); cb.set_label(cbar)
            fig.tight_layout(); fig.savefig(plots_dir / fname, dpi=130); plt.close(fig)

        heatmap("nodes_visited", "m1_heatmap_nodes.png",
                "Среднее число посещённых вершин: алгоритм × топология",
                "log10(вершин)", log=True)
        heatmap("edges_relaxed", "m1_heatmap_edges.png",
                "Среднее число релаксаций рёбер: алгоритм × топология",
                "log10(рёбер)", log=True)
        self.render_fat_tree_example(plots_dir / "fat_tree_example.png")
        tqdm.write(f"Model 1 plots written to {plots_dir}/")

    @staticmethod
    def render_fat_tree_example(path: str | Path, k: int = 4) -> None:
        """Clean layered render of a k-ary fat-tree for the report (Figure 1).

        Independent of the experiment RNG and deliberately small (k=4) for
        readability; uses an explicit core/aggregation/edge/host layout rather
        than a force-directed one so the Clos structure is legible.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
        except Exception:  # pragma: no cover
            return

        half = k // 2
        G = nx.Graph()
        pos: dict = {}
        core = [("core", i) for i in range(half * half)]
        span = k * 4
        for i, c in enumerate(core):
            G.add_node(c)
            pos[c] = ((i + 0.5) * span / len(core), 6)
        for p in range(k):
            base = p * 4
            agg = [("agg", p, i) for i in range(half)]
            edge = [("edge", p, i) for i in range(half)]
            for i, a in enumerate(agg):
                pos[a] = (base + 1 + 2 * i, 4)
            for i, e in enumerate(edge):
                pos[e] = (base + 1 + 2 * i, 2.5)
            for a in agg:
                for e in edge:
                    G.add_edge(a, e)
            for ei, e in enumerate(edge):
                for hh in range(half):
                    host = ("host", p, ei, hh)
                    G.add_edge(e, host)
                    pos[host] = (base + 0.4 + 2 * ei + 1.2 * hh, 1)
            for ai, a in enumerate(agg):
                for c in range(half):
                    G.add_edge(a, core[ai * half + c])

        source = ("host", 0, 0, 0)
        target = ("host", k - 1, half - 1, half - 1)
        colors, sizes = [], []
        for node in G.nodes():
            if node == source:
                colors.append("tab:green"); sizes.append(260)
            elif node == target:
                colors.append("tab:red"); sizes.append(260)
            elif node[0] == "core":
                colors.append("#d98c3f"); sizes.append(200)
            elif node[0] == "agg":
                colors.append("#3b6fb5"); sizes.append(180)
            elif node[0] == "edge":
                colors.append("#5aa469"); sizes.append(160)
            else:
                colors.append("skyblue"); sizes.append(90)

        fig = plt.figure(figsize=(10, 8))
        nx.draw_networkx(G, pos=pos, with_labels=False, node_color=colors,
                         node_size=sizes, edge_color="gray", width=0.9, alpha=0.9)
        for y, label in ((6, "ядро (core)"), (4, "агрегация (aggregation)"),
                         (2.5, "доступ (edge)"), (1, "хосты (hosts)")):
            plt.text(-0.6, y, label, ha="right", va="center", fontsize=9,
                     color="#444444")
        plt.title(f"Топология fat-tree (k={k}): источник — зелёный, получатель — красный")
        plt.axis("off")
        plt.xlim(-4.5, span + 0.5)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close(fig)

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
