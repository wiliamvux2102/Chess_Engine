"""Monte Carlo Tree Search: UCT selection, expansion, rollout, backpropagation."""

import math
import random

import chess
from algorithms.mcts_node import MCTSNode
from algorithms.mcts_rollout import cutoff_value, exchange_gain, position_score, rollout_move


class MCTS:
    """Search a fresh UCT tree per move with fixed iterations and bounded rollouts.

    Simulation rewards use White's perspective: win +1, draw 0, loss -1.
    """

    def __init__(
        self,
        iterations: int = 1000,
        *,
        exploration_constant: float = 2 * math.sqrt(2),
        rollout_max_plies: int = 80,
        seed: int | None = None,
        claim_draw: bool = True,
    ):
        """Configure the iteration budget, UCT exploration and rollout limit.

        rollout_max_plies counts half-moves; zero evaluates without further play.
        seed controls a private RNG; claim_draw should match the controller.
        """
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
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
        self.exploration_constant = exploration_constant
        self.rollout_max_plies = rollout_max_plies
        self.seed = seed
        self.claim_draw = claim_draw
        self._rng = random.Random(seed)
        self.last_search_stats: dict[str, int] = {"iterations": 0, "rollouts": 0}

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose a legal move without changing the input board or its history.

        Run selection, expansion, rollout and backpropagation for each iteration.
        Prefer an immediate win; otherwise choose most visits, then highest mean.
        Reject invalid or terminal positions with ValueError.
        """
        self.last_search_stats = {"iterations": 0, "rollouts": 0}
        if not board.is_valid():
            raise ValueError("Invalid chess position")
        root = MCTSNode(board, claim_draw=self.claim_draw)
        if root.is_terminal():
            raise ValueError("Cannot choose a move in a terminal position")
        winning_moves = self._prepare_root(root)
        for _ in range(self.iterations):
            node = root
            # Selection: descend through fully expanded nodes using UCT.
            while not node.untried_moves and node.children:
                node = node.select_child(self.exploration_constant)
            # Expansion: add one untried move.
            if node.untried_moves:
                move = (node.untried_moves[-1] if node is root else
                        rollout_move(node.board, self._rng, node.untried_moves))
                node = node.expand(self._rng, move)
            value = self._simulate(node.board)
            node.backpropagate_value(value)
        self.last_search_stats = {"iterations": self.iterations, "rollouts": self.iterations}
        if winning_moves:
            return winning_moves[0]
        best = max(root.children, key=lambda child: (child.visits, child.mean_value))
        assert best.move is not None
        return best.move

    def _prepare_root(self, root: MCTSNode) -> list[chess.Move]:
        """Find immediate wins, filter avoidable mate-in-one losses and order moves.

        Remove losing moves only if a safe alternative exists. Highest priority
        is placed last for expansion. Return immediate wins and restore the board.
        """
        board = root.board
        priorities = {move: 0.0 for move in root.untried_moves}
        safe, losing, winning = [], [], []
        for move in list(root.untried_moves):
            gain = exchange_gain(board, move)
            board.push(move)
            try:
                outcome = board.outcome(claim_draw=self.claim_draw)
                if outcome is not None:
                    if outcome.winner == (not board.turn):
                        winning.append(move)
                        priorities[move] = 100000.0
                    else:
                        priorities[move] = 0.0
                    safe.append(move)
                    continue
                score = position_score(board) * (1 if not board.turn else -1)
                if gain < 0:
                    score += gain
                priorities[move] = score
                unsafe = False
                for reply in list(board.legal_moves):
                    if not board.gives_check(reply):
                        continue
                    board.push(reply)
                    try:
                        unsafe = board.is_checkmate()
                    finally:
                        board.pop()
                    if unsafe:
                        break
                if unsafe:
                    losing.append(move)
                    priorities[move] = -100000.0
                else:
                    safe.append(move)
            finally:
                board.pop()
        if safe:
            root.untried_moves = [move for move in root.untried_moves if move not in losing]
        root.untried_moves.sort(key=lambda move: priorities[move])
        return winning

    def _simulate(self, board: chess.Board) -> float:
        """Play a bounded continuation on a copy and return a White-perspective reward.

        Return +1/0/-1 for a terminal result; otherwise use the cutoff estimate
        in [-0.98, +0.98]. Check the outcome after the final permitted ply too.
        """
        rollout = board.copy(stack=True)
        for _ in range(self.rollout_max_plies):
            outcome = rollout.outcome(claim_draw=self.claim_draw)
            if outcome is not None:
                return 0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
            rollout.push(rollout_move(rollout, self._rng))
        # Check again so a mate on the last permitted ply is not lost.
        outcome = rollout.outcome(claim_draw=self.claim_draw)
        if outcome is not None:
            return 0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
        return cutoff_value(rollout)
