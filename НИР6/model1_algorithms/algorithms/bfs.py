from collections import deque

import networkx as nx

from .base import AlgorithmResult, ShortestPathAlgorithm


class BFS(ShortestPathAlgorithm):
    """Shortest-hop (unweighted) BFS.

    Finds the path with the fewest edges, then reports the *weighted*
    cost of that path so it can be compared against optimal algorithms.
    The path itself may be sub-optimal when edge weights differ.
    """

    @property
    def name(self) -> str:
        return "BFS (unweighted)"

    def _run(self, graph: nx.Graph, source: int, target: int) -> AlgorithmResult:
        prev = {source: None}
        visited = {source}
        queue: deque = deque([source])
        edges_relaxed = 0
        found = False

        while queue:
            u = queue.popleft()
            if u == target:
                found = True
                break
            for v in graph[u]:
                edges_relaxed += 1
                if v not in visited:
                    visited.add(v)
                    prev[v] = u
                    queue.append(v)

        if not found:
            return AlgorithmResult(
                self.name, None, float("inf"), 0,
                len(visited), edges_relaxed, success=False,
            )

        path = self._reconstruct_path(prev, target)
        cost = sum(
            graph[path[i]][path[i + 1]].get("weight", 1)
            for i in range(len(path) - 1)
        )
        return AlgorithmResult(
            self.name, path, cost, len(path) - 1,
            len(visited), edges_relaxed,
        )
