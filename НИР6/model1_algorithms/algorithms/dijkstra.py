import heapq

import networkx as nx

from .base import AlgorithmResult, ShortestPathAlgorithm


class Dijkstra(ShortestPathAlgorithm):

    @property
    def name(self) -> str:
        return "Dijkstra"

    def _run(self, graph: nx.Graph, source: int, target: int) -> AlgorithmResult:
        dist = {source: 0}
        prev = {source: None}
        visited: set = set()
        edges_relaxed = 0
        counter = 0
        pq = [(0, counter, source)]

        while pq:
            d, _, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == target:
                break
            for v, data in graph[u].items():
                edges_relaxed += 1
                w = data.get("weight", 1)
                nd = d + w
                if v not in visited and (v not in dist or nd < dist[v]):
                    dist[v] = nd
                    prev[v] = u
                    counter += 1
                    heapq.heappush(pq, (nd, counter, v))

        if target not in visited:
            return AlgorithmResult(
                self.name, None, float("inf"), 0,
                len(visited), edges_relaxed, success=False,
            )

        path = self._reconstruct_path(prev, target)
        return AlgorithmResult(
            self.name, path, dist[target], len(path) - 1,
            len(visited), edges_relaxed,
        )
