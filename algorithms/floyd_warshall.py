import networkx as nx

from .base import AlgorithmResult, ShortestPathAlgorithm


class FloydWarshall(ShortestPathAlgorithm):
    """All-pairs shortest paths via dynamic programming.

    For this benchmark we compute the full distance table, then extract the
    requested source -> target path. The ``nodes_visited`` metric counts how
    many intermediate nodes were processed, while ``edges_relaxed`` counts
    how many distance-improvement checks were attempted.
    """

    @property
    def name(self) -> str:
        return "Floyd-Warshall"

    def _run(self, graph: nx.Graph, source: int, target: int) -> AlgorithmResult:
        nodes = list(graph.nodes())
        dist = {u: {v: float("inf") for v in nodes} for u in nodes}
        nxt = {u: {v: None for v in nodes} for u in nodes}
        edges_relaxed = 0

        for u in nodes:
            dist[u][u] = 0.0
            nxt[u][u] = u

        for u, v, data in graph.edges(data=True):
            w = data.get("weight", 1)
            if w < dist[u][v]:
                dist[u][v] = w
                nxt[u][v] = v
            if not graph.is_directed() and w < dist[v][u]:
                dist[v][u] = w
                nxt[v][u] = u

        nodes_visited = 0
        for k in nodes:
            nodes_visited += 1
            for i in nodes:
                if dist[i][k] == float("inf"):
                    continue
                for j in nodes:
                    edges_relaxed += 1
                    candidate = dist[i][k] + dist[k][j]
                    if candidate < dist[i][j]:
                        dist[i][j] = candidate
                        nxt[i][j] = nxt[i][k]

        if nxt[source][target] is None:
            return AlgorithmResult(
                self.name, None, float("inf"), 0,
                nodes_visited, edges_relaxed, success=False,
            )

        path = [source]
        current = source
        while current != target:
            current = nxt[current][target]
            if current is None:
                return AlgorithmResult(
                    self.name, None, float("inf"), 0,
                    nodes_visited, edges_relaxed, success=False,
                )
            path.append(current)

        return AlgorithmResult(
            self.name, path, dist[source][target], len(path) - 1,
            nodes_visited, edges_relaxed,
        )
