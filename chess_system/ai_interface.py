"""Contract shared by Alpha-Beta, MCTS, GUI, and experiments."""

from typing import Protocol, runtime_checkable

import chess


@runtime_checkable
class ChessAI(Protocol):
    """Return a legal move; restore the supplied board before returning.

    No call is made on a terminal position. Implementations may expose a
    ``last_search_stats`` dict containing depth, nodes, iterations, or rollouts.
    These optional metrics describe only the most recent choose_move call.
    """

    def choose_move(self, board: chess.Board) -> chess.Move:
        ...
