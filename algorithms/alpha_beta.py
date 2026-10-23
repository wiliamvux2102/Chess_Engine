
"""Depth-limited Alpha-Beta Chess AI.

All scores use White's perspective:
White maximizes and Black minimizes.
"""

import chess

from algorithms.evaluation import PIECE_VALUES, evaluate



def move_priority(board: chess.Board, move: chess.Move) -> int:
    """Estimate how promising a move is for search ordering."""
    score = 0

    # 1. Captures: MVV-LVA heuristic
    if board.is_capture(move):
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)

        if board.is_en_passant(move):
            victim_value = PIECE_VALUES[chess.PAWN]
        elif victim is not None:
            victim_value = PIECE_VALUES.get(victim.piece_type, 0)
        else:
            victim_value = 0

        attacker_value = (
            PIECE_VALUES.get(attacker.piece_type, 0)
            if attacker is not None
            else 0
        )

        score += 10_000
        score += 10 * victim_value - attacker_value

    # 2. Pawn promotion
    if move.promotion is not None:
        score += 8_000
        score += PIECE_VALUES.get(move.promotion, 0)

    # 3. Moves giving check
    if board.gives_check(move):
        score += 50

    return score


def order_moves(board: chess.Board) -> list[chess.Move]:
    """Return legal moves sorted by descending priority."""
    return sorted(
        board.legal_moves,
        key=lambda move: move_priority(board, move),
        reverse=True,
    )


class AlphaBetaAI:
    """Chess AI using Alpha-Beta Pruning."""

    def __init__(self, depth: int = 3):
        if depth < 1:
            raise ValueError("Search depth must be positive")

        self.depth = depth
        self.last_search_stats = {
            "depth": 0,
            "nodes": 0,
            "cutoffs": 0,
        }

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose the best legal move without modifying the board."""

        if board.is_game_over(claim_draw=True):
            raise ValueError("Cannot choose a move in a terminal position")

        self.last_search_stats = {
            "depth": 0,
            "nodes": 0,
            "cutoffs": 0,
        }

        maximizing = board.turn == chess.WHITE
        legal_moves = order_moves(board)

        best_move = legal_moves[0]
        best_score = (
            float("-inf") if maximizing else float("inf")
        )

        alpha = float("-inf")
        beta = float("inf")

        for move in legal_moves:
            board.push(move)

            try:
                score = self._alpha_beta(
                    board=board,
                    depth=self.depth - 1,
                    ply=1,
                    alpha=alpha,
                    beta=beta,
                )
            finally:
                board.pop()

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)

            else:
                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)

        self.last_search_stats["depth"] = self.depth

        return best_move

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        ply: int,
        alpha: float,
        beta: float,
    ) -> int:
        """Return a minimax score or an alpha-beta bound."""

        self.last_search_stats["nodes"] += 1

        # 1. Terminal state
        if board.is_game_over(claim_draw=True):
            return evaluate(board, ply=ply)

        # 2. Depth cutoff
        if depth == 0:
            return evaluate(board, ply=ply)

        maximizing = board.turn == chess.WHITE

        # 3. MAX node
        if maximizing:
            best_score = float("-inf")

            for move in order_moves(board):
                board.push(move)

                try:
                    score = self._alpha_beta(
                        board=board,
                        depth=depth - 1,
                        ply=ply + 1,
                        alpha=alpha,
                        beta=beta,
                    )
                finally:
                    board.pop()

                best_score = max(best_score, score)

                alpha = max(alpha, best_score)

                if alpha >= beta:
                    self.last_search_stats["cutoffs"] += 1
                    break

            return best_score

        # 4. MIN node
        else:
            best_score = float("inf")

            for move in order_moves(board):
                board.push(move)

                try:
                    score = self._alpha_beta(
                        board=board,
                        depth=depth - 1,
                        ply=ply + 1,
                        alpha=alpha,
                        beta=beta,
                    )
                finally:
                    board.pop()

                best_score = min(best_score, score)

                beta = min(beta, best_score)

                if alpha >= beta:
                    self.last_search_stats["cutoffs"] += 1
                    break

            return best_score
