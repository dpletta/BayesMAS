"""Deterministic shortest-path lengths for puzzle states."""

from __future__ import annotations

from collections import deque

from bayesmas.puzzles.base import Puzzle


def bfs_distance(puzzle: Puzzle, *, max_nodes: int = 200_000) -> int | None:
    """Breadth-first distance to a solved state, or ``None`` if unreachable.

    Some puzzles (e.g. checker jumping) admit legal moves that lead to
    deadlocked states from which the goal can no longer be reached. Callers
    that need to score such states rather than fail should use this and treat
    ``None`` as "infinitely far from the goal".
    """
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
    return None


def shortest_length(puzzle: Puzzle, *, max_nodes: int = 200_000) -> int:
    """Breadth-first distance from ``puzzle`` to a solved state.

    Raises ``RuntimeError`` when no solution exists. Use :func:`bfs_distance`
    when an unsolvable state should be scored instead of raising.
    """
    dist = bfs_distance(puzzle, max_nodes=max_nodes)
    if dist is None:
        raise RuntimeError(f"no solution for {puzzle.name} state {puzzle.encode()!r}")
    return dist
