import networkx as nx

from .base import AlgorithmResult, ShortestPathAlgorithm


class BellmanFord(ShortestPathAlgorithm):

    @property
    def name(self) -> str:
        return "Bellman-Ford"

    def _run(self, graph: nx.Graph, source: int, target: int) -> AlgorithmResult:
        nodes = list(graph.nodes())
        dist = {n: float("inf") for n in nodes}
        prev: dict = {n: None for n in nodes}
        dist[source] = 0
        edges_relaxed = 0
        nodes_touched: set = set()

        # Materialise directed edge list once (undirected → both directions)
        directed_edges: list[tuple] = []
        for u, v, data in graph.edges(data=True):
            w = data.get("weight", 1)
            directed_edges.append((u, v, w))
            if not graph.is_directed():
                directed_edges.append((v, u, w))

        for _ in range(len(nodes) - 1):
            updated = False
            for u, v, w in directed_edges:
                edges_relaxed += 1
                if dist[u] != float("inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    nodes_touched.add(v)
                    updated = True
            if not updated:
                break

        if dist[target] == float("inf"):
            return AlgorithmResult(
                self.name, None, float("inf"), 0,
                len(nodes_touched), edges_relaxed, success=False,
            )

        path = self._reconstruct_path(prev, target)
        return AlgorithmResult(
            self.name, path, dist[target], len(path) - 1,
            len(nodes_touched), edges_relaxed,
        )
