"""Rule handling and game state, independent of GUI and AI algorithms."""

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class GameResult:
    result: str  # 1-0, 0-1, 1/2-1/2, or * (ongoing)
    winner: chess.Color | None
    termination: str | None


class GameController:
    def __init__(self, fen: str = chess.STARTING_FEN, *, claim_draw: bool = True):
        self._board = chess.Board(fen)
        if not self._board.is_valid():
            raise ValueError("Invalid chess position")
        self.claim_draw = claim_draw

    @property
    def board(self) -> chess.Board:
        """Independent snapshot including move history (needed for repetition)."""
        return self._board.copy(stack=True)

    @property
    def turn(self) -> chess.Color:
        return self._board.turn

    @property
    def legal_moves(self) -> tuple[chess.Move, ...]:
        return tuple(self._board.legal_moves)

    def is_terminal(self) -> bool:
        return self._board.is_game_over(claim_draw=self.claim_draw)

    def result(self) -> GameResult:
        outcome = self._board.outcome(claim_draw=self.claim_draw)
        if outcome is None:
            return GameResult("*", None, None)
        return GameResult(outcome.result(), outcome.winner, outcome.termination.name)

    def push(self, move: chess.Move | str) -> chess.Move:
        """Apply a legal move. Strings use UCI, including promotion: a7a8q."""
        if self.is_terminal():
            raise ValueError("Game has already ended")
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        if move not in self._board.legal_moves:
            raise ValueError(f"Illegal move: {move}")
        self._board.push(move)
        return move

    def pop(self) -> chess.Move:
        """Undo the last move; raise IndexError when there is no move to undo."""
        return self._board.pop()
