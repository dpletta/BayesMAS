"""Deterministic shortest-path lengths for puzzle states."""

from __future__ import annotations

from collections import deque

from bayesmas.puzzles.base import Puzzle


def shortest_length(puzzle: Puzzle, *, max_nodes: int = 200_000) -> int:
    """Breadth-first distance from ``puzzle`` to a solved state."""
    if puzzle.is_solved():
        return 0
    queue: deque[tuple[Puzzle, int]] = deque([(puzzle, 0)])
    seen: set[object] = {puzzle.encode()}
    while queue:
        node, dist = queue.popleft()
        for move in node.legal_moves():
            nxt = node.apply(move)
            key = nxt.encode()
            if key in seen:
                continue
            if nxt.is_solved():
                return dist + 1
            seen.add(key)
            if len(seen) > max_nodes:
                raise RuntimeError(f"BFS exceeded max_nodes={max_nodes}")
            queue.append((nxt, dist + 1))
    raise RuntimeError(f"no solution for {puzzle.name} state {puzzle.encode()!r}")
