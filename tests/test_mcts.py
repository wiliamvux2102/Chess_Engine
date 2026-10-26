import csv
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess

from algorithms.mcts import MCTS
from algorithms.mcts_node import MCTSNode
from experiments.benchmark import run_match
from experiments.mcts_tuning import run_sweep


MATE_IN_ONE = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"


class MCTSNodeTests(unittest.TestCase):
    def test_snapshot_and_expansion_without_duplicate_moves(self):
        board = chess.Board()
        root = MCTSNode(board)
        rng = random.Random(7)
        children = [root.expand(rng) for _ in range(20)]
        self.assertEqual(len({child.move for child in children}), 20)
        self.assertEqual(root.untried_moves, [])
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        self.assertEqual(root.board.fen(), chess.STARTING_FEN)
        for child in children:
            self.assertIs(child.parent, root)
            self.assertIn(child.move, board.legal_moves)
            self.assertEqual(child.board.peek(), child.move)
        with self.assertRaises(ValueError):
            root.expand(rng)

    def test_terminal_node_has_no_expansion(self):
        root = MCTSNode(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"))
        self.assertTrue(root.is_terminal())
        self.assertEqual(root.untried_moves, [])
        with self.assertRaises(ValueError):
            root.select_child(math.sqrt(2))

    def test_uct_exploitation_exploration_and_unvisited(self):
        root = MCTSNode(chess.Board())
        a = root.expand(random.Random(1))
        b = root.expand(random.Random(2))
        self.assertEqual(a.uct_score(1), math.inf)
        root.visits = 100
        a.visits, a.value = 90, 72.0
        b.visits, b.value = 10, 7.0
        self.assertIs(root.select_child(0), a)
        self.assertIs(root.select_child(math.sqrt(2)), b)
        self.assertAlmostEqual(b.uct_score(1), 0.7 + math.sqrt(math.log(100) / 10))

    def test_backpropagation_alternates_player_perspective(self):
        root = MCTSNode(chess.Board())
        child = root.expand(random.Random(1))
        grandchild = child.expand(random.Random(2))
        grandchild.backpropagate(chess.BLACK)
        self.assertEqual([n.visits for n in (root, child, grandchild)], [1, 1, 1])
        self.assertEqual([n.value for n in (root, child, grandchild)], [1, 0, 1])
        grandchild.backpropagate(None)
        self.assertEqual([n.value for n in (root, child, grandchild)], [1.5, 0.5, 1.5])

    def test_opponent_selects_its_own_winning_reply(self):
        root = MCTSNode(chess.Board())
        white_move = root.expand(random.Random(1))
        black_win = white_move.expand(random.Random(2))
        white_win = white_move.expand(random.Random(3))
        black_win.backpropagate(chess.BLACK)
        white_win.backpropagate(chess.WHITE)
        self.assertIs(white_move.select_child(0), black_win)


class MCTSTests(unittest.TestCase):
    def test_invalid_configuration(self):
        invalid = [
            {"iterations": 0}, {"iterations": -1}, {"iterations": 1.5},
            {"iterations": True}, {"iterations": None},
            {"time_limit": 0}, {"time_limit": -1}, {"time_limit": math.inf},
            {"time_limit": math.nan}, {"time_limit": True},
            {"exploration_constant": -1}, {"exploration_constant": math.nan},
            {"exploration_constant": math.inf}, {"exploration_constant": True},
            {"rollout_max_plies": -1}, {"rollout_max_plies": 1.5},
            {"seed": "7"}, {"claim_draw": 1},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MCTS(**kwargs)

    def test_legal_move_input_history_preserved_and_exact_budget(self):
        board = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3"):
            board.push_uci(move)
        before = board.fen(), tuple(board.move_stack)
        ai = MCTS(iterations=30, rollout_max_plies=4, seed=7)
        self.assertIn(ai.choose_move(board), board.legal_moves)
        self.assertEqual((board.fen(), tuple(board.move_stack)), before)
        self.assertEqual(ai.last_search_stats, {"iterations": 30, "rollouts": 30})
        ai.choose_move(board)
        self.assertEqual(ai.last_search_stats["iterations"], 30)

    def test_seeded_search_does_not_change_global_random(self):
        state = random.getstate()
        kwargs = dict(iterations=40, rollout_max_plies=8, seed=42)
        a, b = MCTS(**kwargs), MCTS(**kwargs)
        self.assertEqual(a.choose_move(chess.Board()), b.choose_move(chess.Board()))
        self.assertEqual(random.getstate(), state)

    def test_finds_mate_for_both_colors(self):
        white = chess.Board(MATE_IN_ONE)
        for board in (white, white.mirror()):
            with self.subTest(turn=board.turn):
                ai = MCTS(iterations=160, rollout_max_plies=0, seed=7)
                move = ai.choose_move(board)
                board.push(move)
                self.assertTrue(board.is_checkmate(), move.uci())

    def test_rejects_terminal_and_invalid_positions(self):
        ai = MCTS(iterations=1)
        for fen in (
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",  # Checkmate.
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # Stalemate.
            "7k/8/6K1/8/8/8/8/8 w - - 0 1",  # Insufficient material.
            "8/8/8/8/8/8/8/8 w - - 0 1",  # Invalid.
        ):
            with self.subTest(fen=fen), self.assertRaises(ValueError):
                ai.choose_move(chess.Board(fen))

    def test_claim_draw_policy_keeps_repetition_history(self):
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            board.push_uci(move)
        with self.assertRaises(ValueError):
            MCTS(iterations=1).choose_move(board)
        ai = MCTS(iterations=1, rollout_max_plies=0, claim_draw=False)
        self.assertIn(ai.choose_move(board), board.legal_moves)
        self.assertEqual(len(board.move_stack), 8)

    def test_expired_time_budget_returns_legal_fallback(self):
        ai = MCTS(iterations=None, time_limit=0.1, seed=7)
        with patch("algorithms.mcts.perf_counter", side_effect=[0.0, 1.0]):
            move = ai.choose_move(chess.Board())
        self.assertIn(move, chess.Board().legal_moves)
        self.assertEqual(ai.last_search_stats["iterations"], 0)

    def test_combined_budget_stops_at_iteration_limit(self):
        ai = MCTS(iterations=3, time_limit=1, rollout_max_plies=0)
        with patch("algorithms.mcts.perf_counter", return_value=0):
            ai.choose_move(chess.Board())
        self.assertEqual(ai.last_search_stats["iterations"], 3)

    def test_rollout_terminal_cutoff_and_last_ply_mate(self):
        ai = MCTS(iterations=1, rollout_max_plies=1)
        board = chess.Board(MATE_IN_ONE)
        mate = next(move for move in board.legal_moves if self._is_mate(board, move))
        with patch.object(ai._rng, "choice", return_value=mate):
            self.assertEqual(ai._simulate(board, None), chess.WHITE)
        self.assertEqual(board.fen(), MATE_IN_ONE)
        with patch("algorithms.mcts.perf_counter", return_value=2):
            self.assertIsNone(ai._simulate(board, 1))
        self.assertIsNone(ai._simulate(chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1"), None))

    def test_special_move_positions(self):
        for fen in (
            "7k/P7/8/8/8/8/8/7K w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "7k/8/8/3pP3/8/8/8/7K w - d6 0 1",
        ):
            board = chess.Board(fen)
            ai = MCTS(iterations=8, rollout_max_plies=2, seed=2)
            self.assertIn(ai.choose_move(board), board.legal_moves)
            self.assertEqual(board.fen(), chess.Board(fen).fen())

    def test_works_with_existing_match_runner(self):
        match = run_match(
            MCTS(iterations=4, rollout_max_plies=2, seed=1),
            MCTS(iterations=4, rollout_max_plies=2, seed=2),
            max_plies=4,
        )
        self.assertEqual(len(match.moves), 4)
        self.assertTrue(all(m.iterations == 4 and m.rollouts == 4 for m in match.moves))
        self.assertTrue(all(m.nodes is None and m.depth is None for m in match.moves))

    @staticmethod
    def _is_mate(board, move):
        copy = board.copy()
        copy.push(move)
        return copy.is_checkmate()


class MCTSTuningTests(unittest.TestCase):
    def test_sweep_exports_metrics_and_paired_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep"
            rows = run_sweep(output, constants=[0.5], budgets=[2], seeds=[7],
                             rollout_max_plies=0, matches=True, max_plies=2)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["iterations"] == 2 for row in rows))
            self.assertTrue((output / "metadata.json").is_file())
            with (output / "searches.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            games = output / "c_0_budget_2_seed_7"
            self.assertEqual(len(list(games.glob("*.pgn"))), 2)
            self.assertTrue((games / "standings.csv").is_file())
            with self.assertRaises(FileExistsError):
                run_sweep(output, constants=[0.5], budgets=[2], seeds=[7])

    def test_invalid_sweep_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep"
            for kwargs in ({"constants": []}, {"budgets": [0]}, {"max_plies": 0}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    run_sweep(output, **kwargs)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
