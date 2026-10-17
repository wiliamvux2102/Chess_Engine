import io
import json
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn

from experiments.benchmark import run_match, save_match
from experiments.tournament import run_tournament, summarize


class FirstLegalAI:
    """Test double only, not an implementation of either assigned algorithm."""
    last_search_stats = {"nodes": 1}

    def choose_move(self, board):
        return next(iter(board.legal_moves))


class BenchmarkTests(unittest.TestCase):
    def test_limit_metrics_and_files(self):
        match = run_match(FirstLegalAI(), FirstLegalAI(), max_plies=4)
        self.assertEqual(match.result, "*")
        self.assertEqual(match.termination, "MOVE_LIMIT")
        self.assertEqual(len(match.moves), 4)
        self.assertEqual(match.moves[0].nodes, 1)
        self.assertIsNone(match.moves[0].iterations)
        self.assertTrue(all(move.seconds >= 0 for move in match.moves))
        pgn = chess.pgn.read_game(io.StringIO(match.pgn))
        self.assertEqual(len(list(pgn.mainline_moves())), 4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            save_match(match, directory, "test")
            data = json.loads((Path(directory) / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(data["plies"], 4)
            self.assertTrue((Path(directory) / "test.csv").is_file())
            with self.assertRaises(FileExistsError):
                save_match(match, directory, "test")

    def test_terminal_start_never_calls_ai(self):
        match = run_match(object(), object(), fen="7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(match.result, "1/2-1/2")
        self.assertEqual(match.moves, [])

    def test_checkmate_on_last_allowed_ply(self):
        class ScriptedAI:
            def __init__(self, moves):
                self.moves = iter(moves)

            def choose_move(self, board):
                return chess.Move.from_uci(next(self.moves))

        match = run_match(ScriptedAI(["f2f3", "g2g4"]),
                          ScriptedAI(["e7e5", "d8h4"]), max_plies=4)
        self.assertEqual(match.result, "0-1")
        self.assertEqual(match.termination, "CHECKMATE")

    def test_contract_violations(self):
        class MutatingAI:
            def choose_move(self, board):
                move = next(iter(board.legal_moves))
                board.push(move)
                return move

        class IllegalAI:
            def choose_move(self, board):
                return chess.Move.from_uci("e2e5")

        for agent in (MutatingAI(), IllegalAI()):
            with self.assertRaises(ValueError):
                run_match(agent, FirstLegalAI())

    def test_color_pair_fresh_agents_and_standings(self):
        created = []

        def factory():
            agent = FirstLegalAI()
            created.append(agent)
            return agent

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            output = Path(directory) / "run"
            matches = run_tournament(factory, factory, name_a="A", name_b="B",
                                     max_plies=2, output_dir=output)
            self.assertTrue((output / "standings.csv").is_file())
        self.assertEqual(len(created), 4)
        self.assertEqual([(m.white, m.black) for m in matches], [("A", "B"), ("B", "A")])
        self.assertEqual(matches[0].initial_fen, matches[1].initial_fen)
        self.assertEqual(summarize(matches)["A"]["unfinished"], 2)
        self.assertEqual(summarize(matches)["A"]["draws"], 0)

    def test_win_loss_and_draw_summary(self):
        mate = run_match(FirstLegalAI(), FirstLegalAI(), white_name="A", black_name="B",
                         fen="7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        draw = run_match(FirstLegalAI(), FirstLegalAI(), white_name="A", black_name="B",
                         fen="7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        rows = summarize([mate, draw])
        self.assertEqual(rows["A"]["wins"], 1)
        self.assertEqual(rows["B"]["losses"], 1)
        self.assertEqual(rows["A"]["draws"], 1)


if __name__ == "__main__":
    unittest.main()
