"""Network topology generators with randomised sizes and weights.

Each public method returns ``(graph, source, target)`` where *graph*
is a connected, weighted ``nx.Graph`` with integer node labels.
"""

from __future__ import annotations

import math
from pathlib import Path
import random

import matplotlib.pyplot as plt
import networkx as nx


class TopologyGenerator:
    """Generates diverse graph topologies for algorithm benchmarking.

    Parameters
    ----------
    seed : int
        Master seed — every call advances the internal RNG so that the
        full sequence is reproducible.
    weight_range : tuple[int, int]
        Inclusive (lo, hi) range for random integer edge weights.
    """

    def __init__(self, seed: int = 42, weight_range: tuple[int, int] = (1, 50)):
        self.rng = random.Random(seed)
        self.weight_range = weight_range

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _next_seed(self) -> int:
        return self.rng.randint(0, 2**31)

    def _assign_weights(self, graph: nx.Graph) -> None:
        lo, hi = self.weight_range
        for u, v in graph.edges():
            graph[u][v]["weight"] = self.rng.randint(lo, hi)

    def _pick_endpoints(self, graph: nx.Graph) -> tuple[int, int]:
        nodes = list(graph.nodes())
        return tuple(self.rng.sample(nodes, 2))  # type: ignore[return-value]

    @staticmethod
    def _ensure_connected_ints(graph: nx.Graph) -> nx.Graph:
        """Keep largest component and relabel to 0-based ints."""
        if nx.is_connected(graph):
            return nx.convert_node_labels_to_integers(graph)
        largest_cc = max(nx.connected_components(graph), key=len)
        sub = graph.subgraph(largest_cc).copy()
        return nx.convert_node_labels_to_integers(sub)

    def _finalise(
        self, graph: nx.Graph, *, assign_weights: bool = True
    ) -> tuple[nx.Graph, int, int]:
        graph = self._ensure_connected_ints(graph)
        if assign_weights:
            self._assign_weights(graph)
        s, t = self._pick_endpoints(graph)
        return graph, s, t

    # ------------------------------------------------------------------
    # topologies
    # ------------------------------------------------------------------

    def line(self) -> tuple[nx.Graph, int, int]:
        n = self.rng.randint(20, 300)
        return self._finalise(nx.path_graph(n))

    def ring(self) -> tuple[nx.Graph, int, int]:
        n = self.rng.randint(20, 300)
        return self._finalise(nx.cycle_graph(n))

    def star(self) -> tuple[nx.Graph, int, int]:
        n = self.rng.randint(20, 300)
        return self._finalise(nx.star_graph(n - 1))

    def tree(self) -> tuple[nx.Graph, int, int]:
        h = self.rng.randint(3, 7)
        return self._finalise(nx.balanced_tree(r=2, h=h))

    def grid(self) -> tuple[nx.Graph, int, int]:
        """2-D grid with node positions stored for the A* heuristic."""
        rows = self.rng.randint(4, 20)
        cols = self.rng.randint(4, 20)
        G = nx.grid_2d_graph(rows, cols)
        positions = {(r, c): (float(c), float(r)) for r in range(rows) for c in range(cols)}
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        for old, new in mapping.items():
            G.nodes[new]["pos"] = positions[old]
        self._assign_weights(G)
        s, t = self._pick_endpoints(G)
        return G, s, t

    def dense_mesh(self) -> tuple[nx.Graph, int, int]:
        """Erdos-Renyi graph with p=0.3 (high connectivity)."""
        n = self.rng.randint(20, 150)
        G = nx.erdos_renyi_graph(n, 0.3, seed=self._next_seed())
        return self._finalise(G)

    def sparse_random(self) -> tuple[nx.Graph, int, int]:
        """Erdos-Renyi just above the connectivity threshold."""
        n = self.rng.randint(20, 300)
        p = min(3 * math.log(n) / n, 0.15)
        G = nx.erdos_renyi_graph(n, p, seed=self._next_seed())
        return self._finalise(G)

    def bottleneck(self) -> tuple[nx.Graph, int, int]:
        """Two cliques joined by a single bridge edge."""
        half = self.rng.randint(10, 50)
        G1 = nx.complete_graph(half)
        G2 = nx.complete_graph(range(half, 2 * half))
        G = nx.compose(G1, G2)
        bridge_a = self.rng.randint(0, half - 1)
        bridge_b = self.rng.randint(half, 2 * half - 1)
        G.add_edge(bridge_a, bridge_b)
        self._assign_weights(G)
        # Force endpoints on opposite sides of the bridge
        s = self.rng.randint(0, half - 1)
        t = self.rng.randint(half, 2 * half - 1)
        return G, s, t

    def scale_free(self) -> tuple[nx.Graph, int, int]:
        """Barabasi-Albert preferential-attachment model."""
        n = self.rng.randint(20, 300)
        m = self.rng.randint(2, min(5, n - 1))
        G = nx.barabasi_albert_graph(n, m, seed=self._next_seed())
        return self._finalise(G)

    def small_world(self) -> tuple[nx.Graph, int, int]:
        """Watts-Strogatz small-world graph (k=4, p=0.3)."""
        n = self.rng.randint(20, 300)
        G = nx.watts_strogatz_graph(n, k=4, p=0.3, seed=self._next_seed())
        return self._finalise(G)

    def fat_tree(self) -> tuple[nx.Graph, int, int]:
        """k-ary fat-tree (Clos) — the canonical data-centre topology.

        Three switch tiers (core / aggregation / edge) plus hosts. For pod
        count ``k`` (even): ``(k/2)**2`` core switches, ``k`` pods each with
        ``k/2`` aggregation and ``k/2`` edge switches, and ``k/2`` hosts per
        edge switch. Wiring follows the standard fat-tree (Al-Fares et al.,
        2008): edge<->aggregation is complete bipartite within a pod, and
        aggregation switch ``j`` links to the ``j``-th block of core switches.
        """
        k = self.rng.choice([4, 6, 8])
        half = k // 2
        G = nx.Graph()
        core = [("core", i) for i in range(half * half)]
        G.add_nodes_from(core)
        for pod in range(k):
            agg = [("agg", pod, i) for i in range(half)]
            edge = [("edge", pod, i) for i in range(half)]
            for a in agg:                      # complete bipartite agg<->edge
                for e in edge:
                    G.add_edge(a, e)
            for ei, e in enumerate(edge):      # hosts hang off edge switches
                for hh in range(half):
                    G.add_edge(e, ("host", pod, ei, hh))
            for ai, a in enumerate(agg):       # aggregation <-> core blocks
                for c in range(half):
                    G.add_edge(a, core[ai * half + c])
        return self._finalise(G)

    def dragonfly(self) -> tuple[nx.Graph, int, int]:
        """Dragonfly topology (Kim et al., 2008): groups of routers.

        Each group's ``a`` routers are fully connected (intra-group); groups
        form an all-to-all graph via global links (one link per group pair),
        with ``g = a*h + 1`` groups so every router uses exactly ``h`` global
        ports. ``p`` hosts hang off each router. Balanced choice ``p = h = a/2``.
        """
        a = self.rng.choice([3, 4, 5])
        h = max(1, a // 2)
        p = h
        g = a * h + 1
        G = nx.Graph()

        def R(grp: int, r: int) -> tuple:
            return ("r", grp, r)

        for grp in range(g):                   # intra-group clique + hosts
            routers = [R(grp, r) for r in range(a)]
            for i in range(a):
                for j in range(i + 1, a):
                    G.add_edge(routers[i], routers[j])
            for r in range(a):
                for hp in range(p):
                    G.add_edge(R(grp, r), ("h", grp, r, hp))

        for i in range(g):                     # one global link per group pair
            targets_i = [x for x in range(g) if x != i]
            for j in range(i + 1, g):
                ri = targets_i.index(j) // h
                rj = [x for x in range(g) if x != j].index(i) // h
                G.add_edge(R(i, ri), R(j, rj))
        return self._finalise(G)

    # ------------------------------------------------------------------
    # registry
    # ------------------------------------------------------------------

    def get_all_topologies(self) -> list[tuple[str, callable]]:
        return [
            ("line", self.line),
            ("ring", self.ring),
            ("star", self.star),
            ("tree", self.tree),
            ("grid", self.grid),
            ("dense_mesh", self.dense_mesh),
            ("sparse_random", self.sparse_random),
            ("bottleneck", self.bottleneck),
            ("scale_free", self.scale_free),
            ("small_world", self.small_world),
            ("fat_tree", self.fat_tree),
            ("dragonfly", self.dragonfly),
        ]

    @staticmethod
    def save_graph_image(
        graph: nx.Graph,
        source: int,
        target: int,
        topology: str,
        graph_id: int,
        output_dir: str | Path,
    ) -> Path:
        """Render a graph snapshot with source/target highlighted."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{topology}_graph_{graph_id}.png"

        positions = nx.get_node_attributes(graph, "pos")
        if not positions:
            positions = nx.spring_layout(graph, seed=42)

        node_colors = []
        node_sizes = []
        for node in graph.nodes():
            if node == source:
                node_colors.append("tab:green")
                node_sizes.append(220)
            elif node == target:
                node_colors.append("tab:red")
                node_sizes.append(220)
            else:
                node_colors.append("skyblue")
                node_sizes.append(90)

        plt.figure(figsize=(10, 8))
        nx.draw_networkx(
            graph,
            pos=positions,
            with_labels=False,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color="gray",
            width=0.8,
            alpha=0.9,
        )
        plt.title(f"{topology} graph #{graph_id} | source={source}, target={target}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(image_path, dpi=160)
        plt.close()
        return image_path
