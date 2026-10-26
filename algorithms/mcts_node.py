"""Search-tree nodes for two-player UCT, implemented without an external engine."""

from __future__ import annotations

import math
import random

import chess


class MCTSNode:
    """Own a board snapshot and value from the previous mover's perspective.

    A win is 1, a draw is 0.5, and a loss is 0. Each parent therefore maximizes
    its children's means: the previous mover at a child is the parent's player.
    """

    def __init__(
        self,
        board: chess.Board,
        parent: MCTSNode | None = None,
        move: chess.Move | None = None,
        *,
        claim_draw: bool = True,
    ):
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
        return self.value / self.visits if self.visits else 0.0

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=self.claim_draw)

    def uct_score(self, exploration_constant: float) -> float:
        """Unvisited children are explored before visited children."""
        if self.visits == 0:
            return math.inf
        parent_visits = self.parent.visits if self.parent is not None else self.visits
        return self.mean_value + exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )

    def select_child(self, exploration_constant: float) -> MCTSNode:
        if not self.children:
            raise ValueError("Cannot select a child of a leaf")
        return max(self.children, key=lambda child: child.uct_score(exploration_constant))

    def expand(self, rng: random.Random) -> MCTSNode:
        """Expand exactly one uniformly selected, previously untried legal move."""
        if not self.untried_moves:
            raise ValueError("No untried moves to expand")
        move = self.untried_moves.pop(rng.randrange(len(self.untried_moves)))
        board = self.board.copy(stack=True)
        board.push(move)
        child = MCTSNode(board, self, move, claim_draw=self.claim_draw)
        self.children.append(child)
        return child

    def backpropagate(self, winner: chess.Color | None) -> None:
        """Update the selected node and every ancestor in their own perspective."""
        node: MCTSNode | None = self
        while node is not None:
            node.visits += 1
            node.value += (
                0.5 if winner is None else float(winner == node.player_just_moved)
            )
            node = node.parent
