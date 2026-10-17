import unittest

import chess

from chess_system.game_controller import GameController


class ControllerTests(unittest.TestCase):
    def test_legal_moves_push_pop_and_snapshot(self):
        game = GameController()
        self.assertEqual(len(game.legal_moves), 20)
        snapshot = game.board
        snapshot.push_uci("d2d4")
        self.assertEqual(game.board.fen(), chess.STARTING_FEN)
        game.push("e2e4")
        self.assertEqual(game.turn, chess.BLACK)
        self.assertEqual(game.pop().uci(), "e2e4")
        self.assertEqual(game.board.fen(), chess.STARTING_FEN)
        with self.assertRaises(ValueError):
            game.push("e2e5")

    def test_checkmate(self):
        game = GameController()
        for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
            game.push(move)
        self.assertTrue(game.is_terminal())
        self.assertEqual(game.result().result, "0-1")
        self.assertEqual(game.result().winner, chess.BLACK)
        self.assertEqual(game.result().termination, "CHECKMATE")
        with self.assertRaises(ValueError):
            game.push("a2a3")

    def test_draws(self):
        stalemate = GameController("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(stalemate.result().termination, "STALEMATE")
        insufficient = GameController("7k/8/6K1/8/8/8/8/8 w - - 0 1")
        self.assertEqual(insufficient.result().result, "1/2-1/2")
        game = GameController(claim_draw=False)
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            game.push(move)
        self.assertFalse(game.is_terminal())
        game.claim_draw = True
        self.assertEqual(game.result().termination, "THREEFOLD_REPETITION")
        self.assertTrue(game.board.can_claim_threefold_repetition())

    def test_special_moves(self):
        promotion = GameController("7k/P7/8/8/8/8/8/7K w - - 0 1")
        promotion.push("a7a8q")
        self.assertEqual(promotion.board.piece_at(chess.A8).piece_type, chess.QUEEN)
        castle = GameController("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        castle.push("e1g1")
        self.assertEqual(castle.board.piece_at(chess.F1).piece_type, chess.ROOK)
        game = GameController()
        for move in ("e2e4", "a7a6", "e4e5", "d7d5", "e5d6"):
            game.push(move)
        self.assertIsNone(game.board.piece_at(chess.D5))


if __name__ == "__main__":
    unittest.main()
