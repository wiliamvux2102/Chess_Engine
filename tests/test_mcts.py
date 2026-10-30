"""MCTS search, rollout, evaluation and match-runner integration tests."""

import math
import random
import unittest
from unittest.mock import patch

import chess

from algorithms.mcts import MCTS
from algorithms.mcts_node import MCTSNode
from algorithms.mcts_rollout import PIECE_VALUES, cutoff_value, exchange_gain, position_score, rollout_move
from experiments.benchmark import run_match


MATE_IN_ONE = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"


def tactical_cases():
    """Build labeled tactical puzzles with both original and mirrored colors."""
    defense = chess.Board()
    for move in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4"):
        defense.push_uci(move)
    cases = [
        ("mate", chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"), "mate", None),
        ("free_queen", chess.Board("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"),
         "play", chess.Move.from_uci("d2d5")),
        ("poisoned_rook", chess.Board("4k3/8/4p3/3r4/8/8/3Q4/4K3 w - - 0 1"),
         "avoid", chess.Move.from_uci("d2d5")),
        ("defend_mate", defense, "safe", None),
        ("mate_two_queen", chess.Board("2K5/4Q3/k7/8/8/8/8/8 w - - 0 1"),
         "mate_two", None),
        ("mate_two_king", chess.Board("8/8/Q7/8/8/3K4/8/1k6 w - - 0 1"),
         "mate_two", None),
        ("knight_fork", chess.Board("3q3k/8/8/4N3/8/8/8/R3K3 w - - 0 1"),
         "play", chess.Move.from_uci("e5f7")),
        ("pinned_defender", chess.Board("4k3/8/2n5/1B6/3q4/8/8/3RK3 w - - 0 1"),
         "one_of", (chess.Move.from_uci("d1d4"), chess.Move.from_uci("b5c6"))),
        ("promotion", chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1"),
         "promotion", None),
        ("save_queen", chess.Board("4k2r/8/8/4p3/3Q4/8/P7/4K3 w - - 0 1"),
         "save_queen", None),
    ]
    result = []
    for name, board, goal, move in cases:
        result.append((name + "_original", board, goal, move))
        def mirror(candidate):
            """Mirror a fixture's expected move while preserving promotion."""
            return chess.Move(chess.square_mirror(candidate.from_square),
                              chess.square_mirror(candidate.to_square), promotion=candidate.promotion)
        mirrored_move = (tuple(mirror(candidate) for candidate in move) if isinstance(move, tuple)
                         else mirror(move) if move is not None else None)
        result.append((name + "_mirrored", board.mirror(), goal, mirrored_move))
    return result


def succeeds(board, move, goal, expected):
    """Check a proposed puzzle solution independently of the MCTS evaluator."""
    if goal == "play":
        return move == expected
    if goal == "avoid":
        return move != expected
    if goal == "one_of":
        return move in expected
    if goal == "promotion":
        return move.promotion in (chess.QUEEN, chess.ROOK)
    after = board.copy(stack=True)
    after.push(move)
    if goal == "mate":
        return after.is_checkmate()
    if goal == "mate_two":
        if after.is_checkmate():
            return True
        if after.is_game_over(claim_draw=True):
            return False
        for reply in list(after.legal_moves):
            after.push(reply)
            mate_found = False
            for finish in list(after.legal_moves):
                after.push(finish)
                mate_found = after.is_checkmate()
                after.pop()
                if mate_found:
                    break
            after.pop()
            if not mate_found:
                return False
        return True
    if goal == "save_queen":
        return all(after.piece_type_at(reply.to_square) != chess.QUEEN
                   for reply in after.generate_legal_captures())
    if goal != "safe":
        raise ValueError(f"Unknown puzzle goal: {goal}")
    # A safe defense must not permit an immediate mating reply.
    for reply in list(after.legal_moves):
        after.push(reply)
        mate = after.is_checkmate()
        after.pop()
        if mate:
            return False
    return True


class MCTSPolicyTests(unittest.TestCase):
    def test_exchange_gain_uses_legal_recaptures_and_restores_input(self):
        """Verify exchange estimates and input preservation for both colors."""
        cases = [
            # Endgame material values, normalized with 248 units = 100 cp.
            ("4k3/8/4p3/3r4/8/8/3Q4/4K3 w - - 0 1", "d2d5", (1371 - 2650) * 100 / 248),
            ("4k3/8/2n5/1B6/3q4/8/8/3RK3 w - - 0 1", "d1d4", 2650 * 100 / 248),
            ("7k/8/8/3pP3/8/8/8/7K w - d6 0 1", "e5d6", 100),
            ("3r3k/8/3r4/8/3p4/8/3R4/3R3K w - - 0 1", "d2d4", (248 - 1371) * 100 / 248),
            ("r6k/1P6/8/8/8/8/8/7K w - - 0 1", "b7b8q", -100),
            ("r6k/1P6/8/8/8/8/8/7K w - - 0 1", "b7a8q", (1371 + 2650 - 248) * 100 / 248),
            ("6k1/6p1/8/8/8/8/6Q1/6RK w - - 0 1", "g2g7", 100),
            ("7k/8/4p3/3p4/2P5/8/8/7K w - - 0 1", "c4d5", 0),
        ]
        for fen, uci, expected in cases:
            for mirrored in (False, True):
                with self.subTest(fen=fen, mirrored=mirrored):
                    board, move = chess.Board(fen), chess.Move.from_uci(uci)
                    if mirrored:
                        board = board.mirror()
                        move = chess.Move(chess.square_mirror(move.from_square),
                                          chess.square_mirror(move.to_square), promotion=move.promotion)
                    self.assertTrue(board.is_valid())
                    self.assertIn(move, board.legal_moves)
                    before = board.fen(), tuple(board.move_stack)
                    self.assertAlmostEqual(exchange_gain(board, move), expected)
                    self.assertEqual((board.fen(), tuple(board.move_stack)), before)

    def test_exchange_gain_restores_board_after_exception(self):
        """Verify board restoration when exchange evaluation raises an error."""
        board = chess.Board()
        board.push_uci("e2e4")
        before = board.fen(), tuple(board.move_stack)
        with patch.object(board, "generate_legal_captures", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                exchange_gain(board, chess.Move.from_uci("e7e5"))
        self.assertEqual((board.fen(), tuple(board.move_stack)), before)

    def test_cutoff_accounts_for_immediate_recapture(self):
        """Verify that cutoff evaluation accounts for an immediate recapture."""
        board = chess.Board("4k3/8/4p3/3r4/8/8/3Q4/4K3 w - - 0 1")
        board.push_uci("d2d5")
        self.assertGreater(position_score(board), 0)
        before = board.fen(), tuple(board.move_stack)
        self.assertLess(cutoff_value(board), 0)
        self.assertEqual((board.fen(), tuple(board.move_stack)), before)
        self.assertAlmostEqual(cutoff_value(board.mirror()), -cutoff_value(board))

    def test_rollout_chooses_mate_even_in_uniform_exploration(self):
        """Verify that a proven mate overrides uniform rollout exploration."""
        white = chess.Board(MATE_IN_ONE)
        for board in (white, white.mirror()):
            rng = random.Random(7)
            before = board.fen(), tuple(board.move_stack)
            with patch.object(rng, "random", return_value=0.0):
                move = rollout_move(board, rng)
            self.assertEqual((board.fen(), tuple(board.move_stack)), before)
            board.push(move)
            self.assertTrue(board.is_checkmate())

    def test_candidate_sampling_keeps_only_untried_moves(self):
        """Verify that sampling respects a restricted candidate list."""
        board = chess.Board()
        moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        original = moves.copy()
        rng = random.Random(7)
        for _ in range(10):
            self.assertIn(rollout_move(board, rng, moves), original)
        self.assertEqual(moves, original)
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        with self.assertRaises(ValueError):
            rollout_move(board, rng, [])


class MCTSNodeTests(unittest.TestCase):
    def test_fractional_rewards_preserve_both_players_perspectives(self):
        """Verify signed fractional rewards along the node-to-root path."""
        root = MCTSNode(chess.Board())
        child = root.expand(random.Random(1))
        grandchild = child.expand(random.Random(2))
        grandchild.backpropagate_value(0.73)
        self.assertAlmostEqual(root.value, -0.73)
        self.assertAlmostEqual(child.value, 0.73)
        self.assertAlmostEqual(grandchild.value, -0.73)
        for value in (-1.1, 1.1, math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                child.backpropagate_value(value)
        self.assertEqual(child.visits, 1)

    def test_signed_terminal_rewards_and_draw_for_both_players(self):
        """Verify terminal reward mappings for both player perspectives."""
        for winner, white_value in ((chess.WHITE, 1), (None, 0), (chess.BLACK, -1)):
            with self.subTest(winner=winner):
                root = MCTSNode(chess.Board())
                child = root.expand(random.Random(7))
                child.backpropagate(winner)
                self.assertEqual(child.mean_value, white_value)
                self.assertEqual(root.mean_value, -white_value)

    def test_uct_scaling_preserves_order_after_reward_conversion(self):
        """Verify UCT scaling after conversion to signed rewards."""
        root = MCTSNode(chess.Board())
        a = root.expand(random.Random(1))
        b = root.expand(random.Random(2))
        root.visits = 100
        for child, visits, old_mean in ((a, 90, 0.8), (b, 10, 0.7)):
            child.visits = visits
            child.value = visits * (2 * old_mean - 1)
            old_uct = old_mean + math.sqrt(2) * math.sqrt(math.log(100) / visits)
            self.assertAlmostEqual(child.uct_score(MCTS().exploration_constant), 2 * old_uct - 1)
        self.assertIs(root.select_child(MCTS().exploration_constant), b)

    def test_snapshot_and_expansion_without_duplicate_moves(self):
        """Verify independent snapshots and nonduplicated legal expansions."""
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
        """Verify terminal detection and an empty terminal-node move list."""
        root = MCTSNode(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"))
        self.assertTrue(root.is_terminal())
        self.assertEqual(root.untried_moves, [])
        with self.assertRaises(ValueError):
            root.select_child(math.sqrt(2))

    def test_uct_exploitation_exploration_and_unvisited(self):
        """Verify UCT exploration, exploitation and unvisited-node handling."""
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
        """Verify reward signs and visit counts along alternating players."""
        root = MCTSNode(chess.Board())
        child = root.expand(random.Random(1))
        grandchild = child.expand(random.Random(2))
        grandchild.backpropagate(chess.BLACK)
        self.assertEqual([n.visits for n in (root, child, grandchild)], [1, 1, 1])
        self.assertEqual([n.value for n in (root, child, grandchild)], [1, -1, 1])
        grandchild.backpropagate(None)
        self.assertEqual([n.value for n in (root, child, grandchild)], [1, -1, 1])
        self.assertEqual([n.visits for n in (root, child, grandchild)], [2, 2, 2])

    def test_opponent_selects_its_own_winning_reply(self):
        """Verify that an opponent prefers its own winning reply."""
        root = MCTSNode(chess.Board())
        white_move = root.expand(random.Random(1))
        black_win = white_move.expand(random.Random(2))
        white_win = white_move.expand(random.Random(3))
        black_win.backpropagate(chess.BLACK)
        white_win.backpropagate(chess.WHITE)
        self.assertIs(white_move.select_child(0), black_win)


class MCTSTests(unittest.TestCase):
    def test_cutoff_distinguishes_material_and_is_color_symmetric(self):
        """Verify signed material estimates and symmetry between colors."""
        board = chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
        ai = MCTS(iterations=1, rollout_max_plies=0)
        value = ai._simulate(board)
        self.assertGreater(value, 0)
        self.assertLess(value, 1)
        self.assertAlmostEqual(ai._simulate(board.mirror()), -value)
        self.assertAlmostEqual(position_score(board.mirror()), -position_score(board))
        self.assertEqual(cutoff_value(chess.Board()), 0)

    def test_rollout_returns_signed_terminal_results_at_all_horizons(self):
        """Verify signed terminal results at both rollout-loop boundaries."""
        white_win = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        draw = chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1")
        for horizon in (0, 1):
            ai = MCTS(iterations=1, rollout_max_plies=horizon)
            for board, expected in ((white_win, 1), (white_win.mirror(), -1), (draw, 0)):
                with self.subTest(horizon=horizon, expected=expected):
                    self.assertEqual(ai._simulate(board), expected)

    def test_takes_free_queen_for_both_colors(self):
        """Verify free-queen captures for fixed seeded regression searches."""
        for name, board, goal, expected in tactical_cases():
            if not name.startswith("free_queen"):
                continue
            for seed in (7, 42):
                with self.subTest(case=name, seed=seed):
                    ai = MCTS(iterations=64, rollout_max_plies=12, seed=seed)
                    self.assertEqual(ai.choose_move(board), expected)

    def test_finds_forced_mate_in_two_for_both_colors(self):
        """Verify forced mate-in-two solutions for fixed regression searches."""
        for name, board, goal, expected in tactical_cases():
            if goal == "mate_two":
                with self.subTest(case=name):
                    ai = MCTS(iterations=128, rollout_max_plies=12, seed=19)
                    self.assertTrue(succeeds(board, ai.choose_move(board), goal, expected))

    def test_finds_knight_fork_for_both_colors(self):
        """Verify a knight fork for fixed regression searches of both colors."""
        for name, board, goal, expected in tactical_cases():
            if name.startswith("knight_fork"):
                with self.subTest(case=name):
                    ai = MCTS(iterations=128, rollout_max_plies=12, seed=42)
                    self.assertEqual(ai.choose_move(board), expected)

    def test_root_filter_avoids_immediate_mate_for_both_colors(self):
        """Verify filtering of avoidable immediate mating losses at the root."""
        for name, board, goal, expected in tactical_cases():
            if not name.startswith("defend_mate"):
                continue
            before = board.fen(), tuple(board.move_stack)
            ai = MCTS(iterations=1, rollout_max_plies=0, seed=7)
            root = MCTSNode(board)
            ai._prepare_root(root)
            self.assertTrue(root.untried_moves)
            self.assertLess(len(root.untried_moves), board.legal_moves.count())
            for move in root.untried_moves:
                self.assertTrue(succeeds(board, move, goal, expected), move.uci())
            self.assertTrue(succeeds(board, ai.choose_move(board), goal, expected))
            self.assertEqual((board.fen(), tuple(board.move_stack)), before)

    def test_mate_is_not_discarded_with_one_iteration(self):
        """Verify immediate-mate selection with a one-iteration budget."""
        board = chess.Board(MATE_IN_ONE)
        ai = MCTS(iterations=1, rollout_max_plies=0, seed=19)
        self.assertTrue(self._is_mate(board, ai.choose_move(board)))

    def test_avoids_poisoned_capture(self):
        """Verify avoidance of a labeled losing queen capture."""
        for name, board, goal, expected in tactical_cases():
            if name.startswith("poisoned_rook"):
                ai = MCTS(iterations=64, rollout_max_plies=12, seed=7)
                self.assertNotEqual(ai.choose_move(board), expected)

    def test_root_scan_restores_board_and_history(self):
        """Verify input preservation during root tactical preparation."""
        ai = MCTS(iterations=1, seed=7)
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        root = MCTSNode(board)
        before = root.board.fen(), tuple(root.board.move_stack)
        self.assertEqual(ai._prepare_root(root), [])
        self.assertTrue(root.untried_moves)
        self.assertTrue(all(move in root.board.legal_moves for move in root.untried_moves))
        self.assertEqual((root.board.fen(), tuple(root.board.move_stack)), before)

    def test_invalid_configuration(self):
        """Verify rejection of invalid MCTS constructor settings."""
        invalid = [
            {"iterations": 0}, {"iterations": -1}, {"iterations": 1.5},
            {"iterations": True}, {"iterations": None},
            {"exploration_constant": -1}, {"exploration_constant": math.nan},
            {"exploration_constant": math.inf}, {"exploration_constant": True},
            {"rollout_max_plies": -1}, {"rollout_max_plies": 1.5},
            {"seed": "7"}, {"claim_draw": 1},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MCTS(**kwargs)

    def test_legal_move_input_history_preserved_and_exact_budget(self):
        """Verify legal output, unchanged input and exact per-search metrics."""
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
        """Verify seeded reproducibility and isolation from global randomness."""
        state = random.getstate()
        kwargs = dict(iterations=40, rollout_max_plies=8, seed=42)
        a, b = MCTS(**kwargs), MCTS(**kwargs)
        self.assertEqual(a.choose_move(chess.Board()), b.choose_move(chess.Board()))
        self.assertEqual(random.getstate(), state)

    def test_finds_mate_for_both_colors(self):
        """Verify immediate checkmate selection for both colors."""
        white = chess.Board(MATE_IN_ONE)
        for board in (white, white.mirror()):
            with self.subTest(turn=board.turn):
                ai = MCTS(iterations=160, rollout_max_plies=0, seed=7)
                move = ai.choose_move(board)
                board.push(move)
                self.assertTrue(board.is_checkmate(), move.uci())

    def test_rejects_terminal_and_invalid_positions(self):
        """Verify rejection of terminal and invalid choose_move inputs."""
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
        """Verify draw-claim behavior and repetition-history preservation."""
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            board.push_uci(move)
        with self.assertRaises(ValueError):
            MCTS(iterations=1).choose_move(board)
        ai = MCTS(iterations=1, rollout_max_plies=0, claim_draw=False)
        self.assertIn(ai.choose_move(board), board.legal_moves)
        self.assertEqual(len(board.move_stack), 8)

    def test_fixed_budget_runs_exact_iterations_with_zero_length_rollouts(self):
        """Verify exact iteration counts with zero-length rollouts."""
        ai = MCTS(iterations=3, rollout_max_plies=0)
        ai.choose_move(chess.Board())
        self.assertEqual(ai.last_search_stats, {"iterations": 3, "rollouts": 3})

    def test_rollout_stops_at_ply_limit_and_preserves_input(self):
        """Verify the rollout ply limit and input-board preservation."""
        ai = MCTS(iterations=1, rollout_max_plies=3)
        board = chess.Board()
        expected = board.copy(stack=True)
        for _ in range(3):
            expected.push(next(iter(expected.legal_moves)))
        with patch("algorithms.mcts.rollout_move",
                   side_effect=lambda current, rng: next(iter(current.legal_moves))) as policy:
            self.assertEqual(ai._simulate(board), cutoff_value(expected))
        self.assertEqual(policy.call_count, 3)
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        self.assertEqual(board.move_stack, [])

    def test_rollout_terminal_cutoff_and_last_ply_mate(self):
        """Verify final-ply mates and actual draw rewards in rollouts."""
        ai = MCTS(iterations=1, rollout_max_plies=1)
        board = chess.Board(MATE_IN_ONE)
        mate = next(move for move in board.legal_moves if self._is_mate(board, move))
        with patch("algorithms.mcts.rollout_move", return_value=mate):
            self.assertEqual(ai._simulate(board), 1.0)
        black_board = board.mirror()
        black_mate = chess.Move(chess.square_mirror(mate.from_square), chess.square_mirror(mate.to_square))
        with patch("algorithms.mcts.rollout_move", return_value=black_mate):
            self.assertEqual(ai._simulate(black_board), -1.0)
        self.assertEqual(board.fen(), MATE_IN_ONE)
        self.assertEqual(ai._simulate(chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1")), 0.0)

    def test_special_move_positions(self):
        """Verify legal searches on special-move fixture positions."""
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
        """Verify match-runner integration and MCTS metric compatibility."""
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
        """Check whether a legal move immediately checkmates the opponent."""
        copy = board.copy()
        copy.push(move)
        return copy.is_checkmate()


class MCTSFixtureTests(unittest.TestCase):
    def test_extended_puzzles_have_valid_rules_and_labels(self):
        """Verify fixture validity and accepted tactical solution labels."""
        solutions = {"mate_two_queen": "e7b4", "mate_two_king": "d3c3"}
        for name, board, goal, expected in tactical_cases():
            self.assertTrue(board.is_valid(), name)
            self.assertFalse(board.is_game_over(claim_draw=True), name)
            if goal == "play":
                self.assertIn(expected, board.legal_moves)
            if goal == "one_of":
                self.assertTrue(all(move in board.legal_moves for move in expected))
                self.assertTrue(all(succeeds(board, move, goal, expected) for move in expected))
            if goal == "mate_two":
                move = chess.Move.from_uci(solutions[name.rsplit("_", 1)[0]])
                if name.endswith("mirrored"):
                    move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))
                self.assertIn(move, board.legal_moves)
                self.assertTrue(succeeds(board, move, goal, expected), name)


class MCTSEvaluationTests(unittest.TestCase):
    def test_endgame_score_matches_published_table(self):
        """Verify an exact endgame score against the reference table values."""
        board = chess.Board("4k3/8/8/8/8/8/3N4/4K3 w - - 0 1")
        self.assertTrue(board.is_valid())
        self.assertAlmostEqual(position_score(board), (832 + 9) * 100 / 248)
        self.assertAlmostEqual(PIECE_VALUES[chess.KNIGHT], 832 * 100 / 248)
        self.assertEqual(PIECE_VALUES[chess.PAWN], 100)

    def test_midgame_score_and_promoted_material_clamp(self):
        """Verify full middlegame evaluation and clamping after promotions."""
        for promoted in (False, True):
            with self.subTest(promoted=promoted):
                board = chess.Board()
                if promoted:
                    board.set_piece_at(chess.A2, chess.Piece(chess.QUEEN, chess.WHITE))
                    board.set_piece_at(chess.H7, chess.Piece(chess.QUEEN, chess.BLACK))
                self.assertTrue(board.is_valid())
                self.assertEqual(position_score(board), 0)
                board.push_uci("g1f3")
                self.assertAlmostEqual(position_score(board), 96 * 100 / 248)

    def test_intermediate_phase_blends_both_scores(self):
        """Verify interpolation in a position between the two material limits."""
        board = chess.Board("r3k2r/ppp2ppp/2n5/3q4/8/2N5/PPP2PPP/R2QK2R w KQkq - 0 1")
        self.assertTrue(board.is_valid())
        expected = (-8 * 87 - 46 * 41) / 128 * 100 / 248
        self.assertAlmostEqual(position_score(board), expected)

    def test_king_preference_changes_with_phase(self):
        """Verify that the king table changes its preference across phases."""
        middlegame = chess.Board()
        middlegame.remove_piece_at(chess.E1)
        middlegame.remove_piece_at(chess.G1)
        middlegame.castling_rights = 0
        middlegame.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        center = middlegame.copy()
        center.remove_piece_at(chess.G1)
        center.set_piece_at(chess.D4, chess.Piece(chess.KING, chess.WHITE))
        self.assertTrue(middlegame.is_valid())
        self.assertTrue(center.is_valid())
        self.assertGreater(position_score(middlegame), position_score(center))
        endgame = chess.Board("4k3/p7/8/8/8/8/P7/4K3 w - - 0 1")
        center = endgame.copy()
        center.remove_piece_at(chess.E1)
        center.set_piece_at(chess.D4, chess.Piece(chess.KING, chess.WHITE))
        self.assertTrue(center.is_valid())
        self.assertAlmostEqual(position_score(center) - position_score(endgame),
                               (204 - 112) * 100 / 248)

    def test_playout_scores_preserve_history_and_color_symmetry(self):
        """Verify evaluation invariants on a deterministic legal game sequence."""
        board = chess.Board()
        rng = random.Random(20261030)
        for _ in range(80):
            if board.is_game_over(claim_draw=True):
                break
            before = board.fen(), tuple(board.move_stack)
            score, reward = position_score(board), cutoff_value(board)
            self.assertAlmostEqual(position_score(board.mirror()), -score)
            self.assertAlmostEqual(cutoff_value(board.mirror()), -reward)
            self.assertLessEqual(abs(reward), 0.98)
            self.assertEqual((board.fen(), tuple(board.move_stack)), before)
            board.push(rng.choice(list(board.legal_moves)))


if __name__ == "__main__":
    unittest.main()
