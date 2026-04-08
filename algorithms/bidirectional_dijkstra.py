import heapq

import networkx as nx

from .base import AlgorithmResult, ShortestPathAlgorithm


class BidirectionalDijkstra(ShortestPathAlgorithm):
    """Dijkstra run simultaneously from source and target.

    Always expands the frontier with the smaller minimum distance.
    Terminates when no unexplored path can beat the best known cost.
    """

    @property
    def name(self) -> str:
        return "Bidirectional Dijkstra"

    def _run(self, graph: nx.Graph, source: int, target: int) -> AlgorithmResult:
        FWD, BWD = 0, 1
        dist = [{source: 0}, {target: 0}]
        prev: list[dict] = [{source: None}, {target: None}]
        visited: list[set] = [set(), set()]
        counter = 0
        pq: list[list] = [[(0, counter, source)], []]
        counter += 1
        pq[BWD] = [(0, counter, target)]
        counter += 1

        mu = float("inf")
        best_node = None
        edges_relaxed = 0

        while pq[FWD] and pq[BWD]:
            if pq[FWD][0][0] + pq[BWD][0][0] >= mu:
                break

            side = FWD if pq[FWD][0][0] <= pq[BWD][0][0] else BWD
            other = 1 - side

            d, _, u = heapq.heappop(pq[side])
            if u in visited[side]:
                continue
            visited[side].add(u)

            if u in dist[other]:
                cand = dist[side][u] + dist[other][u]
                if cand < mu:
                    mu = cand
                    best_node = u

            for v, data in graph[u].items():
                edges_relaxed += 1
                w = data.get("weight", 1)
                nd = d + w
                if v not in visited[side] and (
                    v not in dist[side] or nd < dist[side][v]
                ):
                    dist[side][v] = nd
                    prev[side][v] = u
                    counter += 1
                    heapq.heappush(pq[side], (nd, counter, v))

                if v in dist[other]:
                    cand = dist[side].get(v, float("inf")) + dist[other][v]
                    if cand < mu:
                        mu = cand
                        best_node = v

        total_visited = len(visited[FWD]) + len(visited[BWD])

        if best_node is None:
            return AlgorithmResult(
                self.name, None, float("inf"), 0,
                total_visited, edges_relaxed, success=False,
            )

        # Reconstruct: source → best_node → target
        fwd_seg: list = []
        n = best_node
        while n is not None:
            fwd_seg.append(n)
            n = prev[FWD].get(n)
        fwd_seg.reverse()

        bwd_seg: list = []
        n = prev[BWD].get(best_node)
        while n is not None:
            bwd_seg.append(n)
            n = prev[BWD].get(n)

        path = fwd_seg + bwd_seg
        return AlgorithmResult(
            self.name, path, mu, len(path) - 1,
            total_visited, edges_relaxed,
        )
