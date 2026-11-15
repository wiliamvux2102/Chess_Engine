"""Search-tree nodes for two-player UCT, implemented without an external engine."""

from __future__ import annotations

import math
import random

import chess


class MCTSNode:
    """Store a board snapshot and statistics from the previous mover's perspective.

    Each parent can maximize its children's scores for its own side.
    """

    def __init__(
        self,
        board: chess.Board,
        parent: MCTSNode | None = None,
        move: chess.Move | None = None,
        *,
        claim_draw: bool = True,
    ):
        """Copy the board with history and initialize empty search statistics."""
        self.board = board.copy(stack=True)
        self.parent = parent
        self.move = move
        self.claim_draw = claim_draw
        self.player_just_moved = not self.board.turn
        self.visits = 0
        self.value = 0.0
        self.children: list[MCTSNode] = []
        self.untried_moves = (
            [] if self.is_terminal() else list(self.board.legal_moves)
        )

    @property
    def mean_value(self) -> float:
        """Return the previous mover's average reward, or zero before any visits."""
        return self.value / self.visits if self.visits else 0.0

    def is_terminal(self) -> bool:
        """Check whether the game has ended under this node's draw policy."""
        return self.board.is_game_over(claim_draw=self.claim_draw)

    def uct_score(self, exploration_constant: float) -> float:
        """Combine mean reward with the UCT exploration bonus.

        UCT = mean + c * sqrt(log(parent visits) / visits).
        Unvisited nodes receive infinity to encourage exploration.
        """
        if self.visits == 0:
            return math.inf
        parent_visits = self.parent.visits if self.parent is not None else self.visits
        return self.mean_value + exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )

    def select_child(self, exploration_constant: float) -> MCTSNode:
        """Return the child with highest UCT; reject nodes without children."""
        if not self.children:
            raise ValueError("Cannot select a child of a leaf")
        return max(self.children, key=lambda child: child.uct_score(exploration_constant))

    def expand(self, rng: random.Random, move: chess.Move | None = None) -> MCTSNode:
        """Create one child from an untried move without changing the parent board.

        Use the supplied move, or sample uniformly when it is omitted.
        """
        if not self.untried_moves:
            raise ValueError("No untried moves to expand")
        if move is None:
            move = self.untried_moves.pop(rng.randrange(len(self.untried_moves)))
        else:
            self.untried_moves.remove(move)
        board = self.board.copy(stack=True)
        board.push(move)
        child = MCTSNode(board, self, move, claim_draw=self.claim_draw)
        self.children.append(child)
        return child

    def backpropagate(self, winner: chess.Color | None) -> None:
        """Convert a terminal winner into a signed White reward and update the path."""
        self.backpropagate_value(
            0.0 if winner is None else (1.0 if winner == chess.WHITE else -1.0)
        )

    def backpropagate_value(self, white_value: float) -> None:
        """Update this node and its ancestors using a White reward in [-1, 1].

        Increment visits, adding the reward if the previous mover was White and
        its negative if Black. A heuristic zero does not prove a terminal draw.
        """
        if not math.isfinite(white_value) or not -1 <= white_value <= 1:
            raise ValueError("white_value must be finite and between -1 and 1")
        node: MCTSNode | None = self
        while node is not None:
            node.visits += 1
            node.value += white_value if node.player_just_moved else -white_value
            node = node.parent
