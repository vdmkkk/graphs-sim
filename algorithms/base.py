from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx


@dataclass
class AlgorithmResult:
    algorithm: str
    path: list | None
    path_cost: float
    path_hops: int
    nodes_visited: int
    edges_relaxed: int
    execution_time_ms: float = 0.0
    success: bool = True


class ShortestPathAlgorithm(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _run(self, graph: nx.Graph, source: int, target: int) -> AlgorithmResult: ...

    def run(self, graph: nx.Graph, source: int, target: int) -> AlgorithmResult:
        if source == target:
            return AlgorithmResult(self.name, [source], 0, 0, 1, 0, 0.0, True)
        start = time.perf_counter()
        result = self._run(graph, source, target)
        result.execution_time_ms = (time.perf_counter() - start) * 1000
        return result

    @staticmethod
    def _reconstruct_path(prev: dict, target) -> list:
        path: list = []
        node = target
        while node is not None:
            path.append(node)
            node = prev.get(node)
        path.reverse()
        return path
