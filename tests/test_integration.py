"""End-to-end integration tests for GUI, Eval Bar, Live AI vs AI, Controller, and CLI."""

from __future__ import annotations

import io
import json
import tempfile
import time
import tkinter as tk
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import chess

from algorithms.alpha_beta import AlphaBetaAI
from algorithms.mcts import MCTS
from chess_system.ai_interface import ChessAI
from experiments.benchmark import run_match
from experiments.tournament import run_tournament, summarize
from gui.app import ChessGUI, build_agent, compute_eval_bar_state
from main import main


class ProtocolAndEvalBarTests(unittest.TestCase):
    def test_both_ais_satisfy_protocol(self):
        ab = AlphaBetaAI(depth=1)
        mcts = MCTS(iterations=5, rollout_max_plies=4, seed=7)
        self.assertIsInstance(ab, ChessAI)
        self.assertIsInstance(mcts, ChessAI)

    def test_build_agent_validation(self):
        self.assertIsNone(build_agent("human"))
        ab = build_agent("alpha-beta", depth="2")
        self.assertIsInstance(ab, AlphaBetaAI)
        self.assertEqual(ab.depth, 2)

        mcts = build_agent("mcts", iterations="20", time_limit="0.5", seed=1)
        self.assertIsInstance(mcts, MCTS)
        self.assertEqual(mcts.iterations, 20)
        self.assertEqual(mcts.time_limit, 0.5)

        mcts_time_only = build_agent("mcts", iterations="", time_limit="0.2")
        self.assertIsInstance(mcts_time_only, MCTS)
        self.assertIsNone(mcts_time_only.iterations)
        self.assertEqual(mcts_time_only.time_limit, 0.2)

        for bad_kwargs in (
            {"kind": "unknown"},
            {"kind": "alpha_beta", "depth": "0"},
            {"kind": "alpha_beta", "depth": "abc"},
            {"kind": "mcts", "iterations": "", "time_limit": ""},
            {"kind": "mcts", "iterations": "-5"},
            {"kind": "mcts", "iterations": "10", "time_limit": "0"},
        ):
            with self.subTest(bad_kwargs=bad_kwargs), self.assertRaises(ValueError):
                build_agent(**bad_kwargs)

    def test_eval_bar_state_across_positions(self):
        # 1. Equal starting position
        ratio, label, cp = compute_eval_bar_state(chess.Board())
        self.assertAlmostEqual(ratio, 0.5, places=2)
        self.assertEqual(label, "0.0")
        self.assertEqual(cp, 0)

        # 2. White up a Queen
        white_up = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        w_ratio, w_label, w_cp = compute_eval_bar_state(white_up)
        self.assertGreater(w_ratio, 0.8)
        self.assertTrue(w_label.startswith("+"))
        self.assertGreater(w_cp, 800)

        # 3. Black up a Queen
        black_up = chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1")
        b_ratio, b_label, b_cp = compute_eval_bar_state(black_up)
        self.assertLess(b_ratio, 0.2)
        self.assertTrue(b_label.startswith("-"))
        self.assertLess(b_cp, -800)

        # 4. White checkmate
        white_mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        m_ratio, m_label, _ = compute_eval_bar_state(white_mate)
        self.assertEqual(m_ratio, 1.0)
        self.assertEqual(m_label, "1-0")

        # 5. Stalemate draw
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        d_ratio, d_label, d_cp = compute_eval_bar_state(stalemate)
        self.assertEqual(d_ratio, 0.5)
        self.assertEqual(d_label, "0.0")
        self.assertEqual(d_cp, 0)


class GUIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.errors: list[tuple[str, str]] = []
        self.gui: ChessGUI | None = None

    def tearDown(self):
        if self.gui is not None:
            self.gui.close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _wait_for_ai_move(self, gui: ChessGUI, timeout: float = 5.0) -> bool:
        initial_plies = len(gui.controller.board.move_stack)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self.root.update()
            gui.drain_ai_queue()
            if len(gui.controller.board.move_stack) > initial_plies:
                return True
            time.sleep(0.01)
        return False

    def test_eval_bar_toggle_and_flip(self):
        gui = self.gui = ChessGUI(
            self.root,
            white_kind="human",
            black_kind="human",
            show_eval_bar=True,
            error_handler=lambda t, m: self.errors.append((t, m)),
        )
        self.assertTrue(gui.show_eval_var.get())
        self.assertTrue(bool(gui.eval_canvas.grid_info()))

        gui.toggle_eval_bar(False)
        self.assertFalse(gui.show_eval_var.get())
        self.assertEqual(gui.eval_canvas.grid_info(), {})

        gui.toggle_eval_bar(True)
        self.assertTrue(gui.show_eval_var.get())
        self.assertTrue(bool(gui.eval_canvas.grid_info()))

        # Flipping board preserves squares bijection and redraws cleanly
        self.assertEqual(gui.square_to_grid(chess.A1), (0, 7))
        gui.flip_board()
        self.assertTrue(gui.flipped)
        self.assertEqual(gui.square_to_grid(chess.A1), (7, 0))

    def test_human_move_legal_highlights_and_interaction_errors(self):
        gui = self.gui = ChessGUI(
            self.root,
            white_kind="human",
            black_kind="human",
            error_handler=lambda t, m: self.errors.append((t, m)),
        )
        # Undo on empty stack returns False safely
        self.assertFalse(gui.undo_move())

        # Clicking opponent piece (Black E7) on White's turn does nothing
        self.assertFalse(gui.select_square(chess.E7))
        self.assertIsNone(gui.selected_square)

        # Clicking White E2 selects it and populates E3, E4 as legal targets
        self.assertFalse(gui.select_square(chess.E2))
        self.assertEqual(gui.selected_square, chess.E2)
        self.assertEqual(set(gui.legal_targets.keys()), {chess.E3, chess.E4})

        # Clicking illegal destination E5 clears selection without moving
        self.assertFalse(gui.select_square(chess.E5))
        self.assertIsNone(gui.selected_square)
        self.assertEqual(len(gui.controller.board.move_stack), 0)

        # Select E2 and play E4
        gui.select_square(chess.E2)
        self.assertTrue(gui.select_square(chess.E4))
        self.assertEqual(gui.controller.board.peek(), chess.Move.from_uci("e2e4"))
        self.assertEqual(len(gui.move_log), 1)

        # Undo restores starting position
        self.assertTrue(gui.undo_move())
        self.assertEqual(gui.controller.board.fen(), chess.STARTING_FEN)

    def test_pawn_promotion_handling(self):
        promo_fen = "7k/P7/8/8/8/8/8/7K w - - 0 1"
        chosen_piece = chess.KNIGHT
        gui = self.gui = ChessGUI(
            self.root,
            fen=promo_fen,
            white_kind="human",
            black_kind="human",
            promotion_chooser=lambda color: chosen_piece,
            error_handler=lambda t, m: self.errors.append((t, m)),
        )
        gui.select_square(chess.A7)
        self.assertIn(chess.A8, gui.legal_targets)
        self.assertEqual(len(gui.legal_targets[chess.A8]), 4)

        self.assertTrue(gui.select_square(chess.A8))
        promoted = gui.controller.board.piece_at(chess.A8)
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted.piece_type, chess.KNIGHT)
        self.assertEqual(gui.controller.board.peek().uci(), "a7a8n")

    def test_human_vs_ai_and_stale_search_cancellation(self):
        gui = self.gui = ChessGUI(
            self.root,
            white_kind="human",
            black_kind="alpha_beta",
            black_depth=1,
            move_delay_ms=5,
            error_handler=lambda t, m: self.errors.append((t, m)),
        )
        # Human plays e2e4 -> triggers Black Alpha-Beta
        gui.select_square(chess.E2)
        self.assertTrue(gui.select_square(chess.E4))
        self.assertEqual(gui.controller.turn, chess.BLACK)

        # Clicking board during AI's turn must be rejected
        self.assertFalse(gui.select_square(chess.E7))

        # Wait for Black AI to respond
        self.assertTrue(self._wait_for_ai_move(gui))
        self.assertEqual(len(gui.controller.board.move_stack), 2)
        self.assertEqual(gui.controller.turn, chess.WHITE)
        self.assertIn("depth=1", gui.stats_var.get())

        # Inject a stale AI move with an old search_token and verify it is ignored
        stale_token = gui.search_token - 1
        gui.ai_queue.put((stale_token, chess.Move.from_uci("d2d4"), 0.01, {}, None))
        self.assertFalse(gui.drain_ai_queue())
        self.assertEqual(len(gui.controller.board.move_stack), 2)

    def test_live_ai_vs_ai_step_and_autoplay(self):
        gui = self.gui = ChessGUI(
            self.root,
            white_kind="alpha_beta",
            black_kind="mcts",
            white_depth=1,
            black_iterations=8,
            move_delay_ms=5,
            auto_start=False,
            error_handler=lambda t, m: self.errors.append((t, m)),
        )
        self.assertTrue(gui.paused)
        self.assertEqual(len(gui.controller.board.move_stack), 0)

        # Step 1: White (Alpha-Beta) plays 1 move and stays paused
        self.assertTrue(gui.step_ai_move())
        self.assertTrue(self._wait_for_ai_move(gui))
        self.assertEqual(len(gui.controller.board.move_stack), 1)
        self.assertTrue(gui.paused)

        # Step 2: Black (MCTS) plays 1 move and stays paused
        self.assertTrue(gui.step_ai_move())
        self.assertTrue(self._wait_for_ai_move(gui))
        self.assertEqual(len(gui.controller.board.move_stack), 2)
        self.assertIn("iterations=8", gui.stats_var.get())

        # Resume live auto-play and watch at least 2 more plies play out on the board
        gui.toggle_pause()
        self.assertFalse(gui.paused)
        self.assertTrue(self._wait_for_ai_move(gui))
        self.assertTrue(self._wait_for_ai_move(gui))
        self.assertGreaterEqual(len(gui.controller.board.move_stack), 4)
        gui.toggle_pause()

    def test_invalid_config_handled_gracefully(self):
        gui = self.gui = ChessGUI(
            self.root,
            white_kind="human",
            black_kind="human",
            error_handler=lambda t, m: self.errors.append((t, m)),
        )
        gui.white_kind_var.set("alpha_beta")
        gui.white_depth_var.set("0")
        self.assertFalse(gui.new_game())
        self.assertTrue(len(self.errors) >= 1)
        self.assertIsNotNone(gui.last_error)


class BenchmarkAndCLIIntegrationTests(unittest.TestCase):
    def test_alpha_beta_vs_mcts_match_and_tournament(self):
        match = run_match(
            AlphaBetaAI(depth=1),
            MCTS(iterations=8, rollout_max_plies=4, seed=7),
            white_name="Alpha-Beta",
            black_name="MCTS",
            max_plies=4,
        )
        self.assertEqual(len(match.moves), 4)
        self.assertEqual(match.moves[0].depth, 1)
        self.assertIsNotNone(match.moves[0].nodes)
        self.assertEqual(match.moves[1].iterations, 8)
        self.assertEqual(match.moves[1].rollouts, 8)

        matches = run_tournament(
            factory_a=lambda: AlphaBetaAI(depth=1),
            factory_b=lambda: MCTS(iterations=6, rollout_max_plies=4, seed=11),
            name_a="Alpha-Beta",
            name_b="MCTS",
            rounds=1,
            max_plies=2,
        )
        standings = summarize(matches)
        self.assertEqual(standings["Alpha-Beta"]["games"], 2)
        self.assertEqual(standings["MCTS"]["games"], 2)

    def test_main_cli_match_and_tournament(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            match_dir = Path(tmpdir) / "match_out"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "match",
                        "--white",
                        "alpha-beta",
                        "--white-depth",
                        "1",
                        "--black",
                        "mcts",
                        "--black-iterations",
                        "6",
                        "--max-plies",
                        "2",
                        "--output-dir",
                        str(match_dir),
                        "--match-id",
                        "demo",
                    ]
                )
            self.assertEqual(rc, 0)
            summary_data = json.loads(buf.getvalue())
            self.assertEqual(summary_data["plies"], 2)
            self.assertTrue((match_dir / "demo.pgn").is_file())
            self.assertTrue((match_dir / "demo.csv").is_file())
            self.assertTrue((match_dir / "demo.json").is_file())

            tourney_dir = Path(tmpdir) / "tourney_out"
            buf_t = io.StringIO()
            with redirect_stdout(buf_t):
                rc_t = main(
                    [
                        "tournament",
                        "--depth",
                        "1",
                        "--iterations",
                        "6",
                        "--rounds",
                        "1",
                        "--max-plies",
                        "2",
                        "--output-dir",
                        str(tourney_dir),
                    ]
                )
            self.assertEqual(rc_t, 0)
            standings = json.loads(buf_t.getvalue())
            self.assertIn("Alpha-Beta", standings)
            self.assertIn("MCTS", standings)
            self.assertTrue((tourney_dir / "standings.csv").is_file())


if __name__ == "__main__":
    unittest.main()
