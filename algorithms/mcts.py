"""Monte Carlo Tree Search: UCT selection, expansion, rollout, backpropagation."""

import math
import random
from time import perf_counter

import chess
from algorithms.mcts_node import MCTSNode


class MCTS:
    """Budgeted MCTS compatible with chess_system.ai_interface.ChessAI.

    Rollouts sample legal moves uniformly. Nonterminal rollouts stopped by the
    depth or time limit receive a neutral 0.5 estimate, not an actual game draw.
    If both budgets are supplied, stop at whichever is reached first. Time is a
    cooperative deadline, not a hard process timeout. Seeded iteration-only
    searches are reproducible; wall-time searches need not be.
    """

    def __init__(
        self,
        iterations: int | None = 1000,
        *,
        time_limit: float | None = None,
        exploration_constant: float = math.sqrt(2),
        rollout_max_plies: int = 80,
        seed: int | None = None,
        claim_draw: bool = True,
    ):
        if iterations is not None and (type(iterations) is not int or iterations < 1):
            raise ValueError("iterations must be a positive integer or None")
        if time_limit is not None and (
            isinstance(time_limit, bool) or not isinstance(time_limit, (int, float))
            or not math.isfinite(time_limit) or time_limit <= 0
        ):
            raise ValueError("time_limit must be a finite positive number or None")
        if iterations is None and time_limit is None:
            raise ValueError("At least one search budget is required")
        if (
            isinstance(exploration_constant, bool)
            or not isinstance(exploration_constant, (int, float))
            or not math.isfinite(exploration_constant) or exploration_constant < 0
        ):
            raise ValueError("exploration_constant must be finite and nonnegative")
        if type(rollout_max_plies) is not int or rollout_max_plies < 0:
            raise ValueError("rollout_max_plies must be a nonnegative integer")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        if type(claim_draw) is not bool:
            raise ValueError("claim_draw must be a bool")
        self.iterations = iterations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.rollout_max_plies = rollout_max_plies
        self.seed = seed
        self.claim_draw = claim_draw
        self._rng = random.Random(seed)
        self.last_search_stats: dict[str, int] = {"iterations": 0, "rollouts": 0}

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Search a private tree; never push or pop on the caller's board."""
        start = perf_counter()
        deadline = None if self.time_limit is None else start + self.time_limit
        self.last_search_stats = {"iterations": 0, "rollouts": 0}
        if not board.is_valid():
            raise ValueError("Invalid chess position")
        root = MCTSNode(board, claim_draw=self.claim_draw)
        if root.is_terminal():
            raise ValueError("Cannot choose a move in a terminal position")
        # Even a deadline exhausted during setup must return a legal move.
        fallback = self._rng.choice(root.untried_moves)
        completed = 0
        while self.iterations is None or completed < self.iterations:
            if deadline is not None and perf_counter() >= deadline:
                break
            node = root
            # Selection: descend through fully expanded nodes using UCT.
            while not node.untried_moves and node.children:
                if deadline is not None and perf_counter() >= deadline:
                    break
                node = node.select_child(self.exploration_constant)
            if deadline is not None and perf_counter() >= deadline:
                break
            # Expansion: terminal nodes have no untried moves.
            if node.untried_moves:
                node = node.expand(self._rng)
            winner = self._simulate(node.board, deadline)
            node.backpropagate(winner)
            completed += 1
        self.last_search_stats = {"iterations": completed, "rollouts": completed}
        if not root.children:
            return fallback
        # Robust child: visits first, empirical mean as the tie-breaker.
        best = max(root.children, key=lambda child: (child.visits, child.mean_value))
        assert best.move is not None
        return best.move

    def _simulate(self, board: chess.Board, deadline: float | None) -> chess.Color | None:
        rollout = board.copy(stack=True)
        for _ in range(self.rollout_max_plies):
            outcome = rollout.outcome(claim_draw=self.claim_draw)
            if outcome is not None:
                return outcome.winner
            if deadline is not None and perf_counter() >= deadline:
                return None
            rollout.push(self._rng.choice(list(rollout.legal_moves)))
        # Check again so a mate on the last permitted ply is not lost.
        outcome = rollout.outcome(claim_draw=self.claim_draw)
        return outcome.winner if outcome is not None else None
