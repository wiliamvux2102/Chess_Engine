
"""Tests for Alpha-Beta Chess AI."""

import unittest

import chess

from algorithms.alpha_beta import AlphaBetaAI, order_moves
from algorithms.evaluation import (
    MATE_SCORE,
    evaluate,
    evaluate_material,
    evaluate_position,
    evaluate_terminal,
)
from experiments.benchmark import run_match


def reference_minimax(
    board: chess.Board,
    depth: int,
    ply: int,
) -> int:
    """Full Minimax without Alpha-Beta pruning."""

    if board.is_game_over(claim_draw=True) or depth == 0:
        return evaluate(board, ply=ply)

    scores = []

    for move in list(board.legal_moves):
        board.push(move)

        try:
            score = reference_minimax(
                board,
                depth - 1,
                ply + 1,
            )
        finally:
            board.pop()

        scores.append(score)

    if board.turn == chess.WHITE:
        return max(scores)

    return min(scores)


def reference_best_score(
    board: chess.Board,
    depth: int,
) -> int:
    """Compute the exact root score using full Minimax."""
    scores = []

    for move in list(board.legal_moves):
        board.push(move)

        try:
            score = reference_minimax(
                board,
                depth - 1,
                1,
            )
        finally:
            board.pop()

        scores.append(score)

    if board.turn == chess.WHITE:
        return max(scores)

    return min(scores)


class EvaluationTests(unittest.TestCase):

    def test_starting_position(self):
        board = chess.Board()

        self.assertEqual(evaluate_material(board), 0)
        self.assertEqual(evaluate_position(board), 0)
        self.assertEqual(evaluate(board), 0)

    def test_material_advantage(self):
        board = chess.Board(
            "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
        )

        self.assertEqual(evaluate_material(board), 900)

    def test_black_checkmate_win(self):
        board = chess.Board()

        for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
            board.push_uci(move)

        self.assertTrue(board.is_checkmate())
        self.assertEqual(
            evaluate_terminal(board, ply=4),
            -MATE_SCORE + 4,
        )

    def test_stalemate(self):
        board = chess.Board(
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
        )

        self.assertTrue(board.is_stalemate())
        self.assertEqual(evaluate(board), 0)


class AlphaBetaTests(unittest.TestCase):

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            AlphaBetaAI(depth=0)

    def test_legal_move_and_board_preservation(self):
        board = chess.Board()
        ai = AlphaBetaAI(depth=2)

        before = (
            board.fen(),
            tuple(board.move_stack),
        )

        move = ai.choose_move(board)

        self.assertIn(move, board.legal_moves)

        after = (
            board.fen(),
            tuple(board.move_stack),
        )

        self.assertEqual(before, after)

    def test_black_can_choose_move(self):
        board = chess.Board()
        board.push_uci("e2e4")

        ai = AlphaBetaAI(depth=2)
        move = ai.choose_move(board)

        self.assertIn(move, board.legal_moves)

    def test_terminal_position_rejected(self):
        board = chess.Board(
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
        )

        ai = AlphaBetaAI(depth=2)

        with self.assertRaises(ValueError):
            ai.choose_move(board)

    def test_move_ordering_preserves_legal_moves(self):
        board = chess.Board()

        original = list(board.legal_moves)
        ordered = order_moves(board)

        self.assertEqual(len(original), len(ordered))
        self.assertEqual(set(original), set(ordered))

    def test_alpha_beta_matches_minimax(self):
        positions = [
            chess.Board(),
            chess.Board(
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/"
                "PPPP1PPP/RNBQKBNR w KQkq - 0 2"
            ),
        ]

        for board in positions:
            with self.subTest(fen=board.fen()):
                depth = 2

                expected = reference_best_score(
                    board,
                    depth,
                )

                ai = AlphaBetaAI(depth=depth)
                move = ai.choose_move(board)

                board.push(move)

                try:
                    actual = reference_minimax(
                        board,
                        depth - 1,
                        1,
                    )
                finally:
                    board.pop()

                self.assertEqual(expected, actual)

    def test_search_stats(self):
        board = chess.Board()
        ai = AlphaBetaAI(depth=2)

        ai.choose_move(board)

        stats = ai.last_search_stats

        self.assertEqual(stats["depth"], 2)
        self.assertGreater(stats["nodes"], 0)
        self.assertGreaterEqual(stats["cutoffs"], 0)

    def test_benchmark_integration(self):
        match = run_match(
            AlphaBetaAI(depth=1),
            AlphaBetaAI(depth=1),
            max_plies=2,
        )

        self.assertEqual(len(match.moves), 2)
        self.assertTrue(
            all(record.nodes is not None
                for record in match.moves)
        )


if __name__ == "__main__":
    unittest.main()
