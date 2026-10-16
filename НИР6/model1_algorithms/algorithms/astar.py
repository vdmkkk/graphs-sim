import heapq
import math

import networkx as nx

from .base import AlgorithmResult, ShortestPathAlgorithm


class AStar(ShortestPathAlgorithm):
    """A* with Euclidean heuristic when node positions are available.

    The heuristic is admissible for grid topologies where every edge
    weight >= 1 and adjacent nodes are unit-distance apart.  For graphs
    without ``pos`` node attributes the heuristic falls back to 0
    (equivalent to Dijkstra).
    """

    @property
    def name(self) -> str:
        return "A*"

    @staticmethod
    def _heuristic(graph: nx.Graph, node: int, target: int) -> float:
        pos_a = graph.nodes[node].get("pos")
        pos_b = graph.nodes[target].get("pos")
        if pos_a is not None and pos_b is not None:
            return math.dist(pos_a, pos_b)
        return 0.0

    def _run(self, graph: nx.Graph, source: int, target: int) -> AlgorithmResult:
        g_score = {source: 0}
        prev = {source: None}
        closed: set = set()
        edges_relaxed = 0
        counter = 0
        h0 = self._heuristic(graph, source, target)
        pq = [(h0, counter, source)]

        while pq:
            _, _, u = heapq.heappop(pq)
            if u in closed:
                continue
            closed.add(u)
            if u == target:
                break
            g_u = g_score[u]
            for v, data in graph[u].items():
                edges_relaxed += 1
                w = data.get("weight", 1)
                tentative = g_u + w
                if v not in closed and (v not in g_score or tentative < g_score[v]):
                    g_score[v] = tentative
                    prev[v] = u
                    counter += 1
                    f_v = tentative + self._heuristic(graph, v, target)
                    heapq.heappush(pq, (f_v, counter, v))

        if target not in closed:
            return AlgorithmResult(
                self.name, None, float("inf"), 0,
                len(closed), edges_relaxed, success=False,
            )

        path = self._reconstruct_path(prev, target)
        return AlgorithmResult(
            self.name, path, g_score[target], len(path) - 1,
            len(closed), edges_relaxed,
        )
