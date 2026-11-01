"""Unified evaluation registry for Alpha-Beta and MCTS.

Allows switching between different evaluation functions:
- 'ab_classic': Original Alpha-Beta evaluation (simple PST + material)
- 'mcts_pe': MCTS PeSTO-style evaluation (tapered eval, SEE, tapered PSQT)
- 'hybrid': MCTS evaluation with Alpha-Beta's terminal handling
"""

import math
from enum import Enum
from typing import Callable

import chess

from algorithms.evaluation import evaluate as ab_evaluate, MATE_SCORE as AB_MATE_SCORE
from algorithms.mcts_rollout import position_score as mcts_position_score, cutoff_value as mcts_cutoff_value
from algorithms.mcts_rollout import CENTIPAWN_SCALE


class EvalMode(Enum):
    AB_CLASSIC = "ab_classic"
    MCTS_PE = "mcts_pe"
    HYBRID = "hybrid"


def create_evaluator(mode: EvalMode, claim_draw: bool = True) -> Callable[[chess.Board, int], float]:
    """Return an evaluation function matching the specified mode.

    All evaluators return score from White's perspective in centipawns.
    For MCTS modes, the score is converted to reward in [-1, 1] via tanh.
    """
    if mode == EvalMode.AB_CLASSIC:
        def evaluator(board: chess.Board, ply: int = 0) -> float:
            return ab_evaluate(board, ply=ply, claim_draw=claim_draw)
        return evaluator

    elif mode == EvalMode.MCTS_PE:
        def evaluator(board: chess.Board, ply: int = 0) -> float:
            # MCTS position_score returns centipawns, convert to reward
            cp = mcts_position_score(board)
            # Check terminal
            outcome = board.outcome(claim_draw=claim_draw)
            if outcome is not None:
                if outcome.winner == chess.WHITE:
                    return 1.0
                elif outcome.winner == chess.BLACK:
                    return -1.0
                return 0.0
            # Scale to [-1, 1] via tanh (same as cutoff_value)
            if abs(cp) >= 10000:  # Near mate
                return 0.98 if cp > 0 else -0.98
            return 0.98 * math.tanh(cp / 600.0)
        return evaluator

    elif mode == EvalMode.HYBRID:
        # Use Alpha-Beta's terminal + material/positional but MCTS's scaling
        def evaluator(board: chess.Board, ply: int = 0) -> float:
            # Terminal check
            terminal = ab_evaluate(board, ply=ply, claim_draw=claim_draw)
            if abs(terminal) >= AB_MATE_SCORE - 1000:
                return 1.0 if terminal > 0 else -1.0
            # Use Alpha-Beta evaluation (centipawns) with MCTS scaling
            cp = ab_evaluate(board, ply=ply, claim_draw=claim_draw)
            if abs(cp) >= 10000:
                return 0.98 if cp > 0 else -0.98
            return 0.98 * math.tanh(cp / 600.0)
        return evaluator

    raise ValueError(f"Unknown eval mode: {mode}")


def create_cutoff_evaluator(mode: EvalMode, claim_draw: bool = True) -> Callable[[chess.Board], float]:
    """Return a cutoff evaluator for MCTS rollout termination."""
    if mode == EvalMode.AB_CLASSIC:
        def evaluator(board: chess.Board) -> float:
            cp = ab_evaluate(board, claim_draw=claim_draw)
            if abs(cp) >= AB_MATE_SCORE - 1000:
                return 1.0 if cp > 0 else -1.0
            return math.tanh(cp / 600.0)
        return evaluator

    elif mode == EvalMode.MCTS_PE:
        return lambda board: mcts_cutoff_value(board)

    elif mode == EvalMode.HYBRID:
        def evaluator(board: chess.Board) -> float:
            cp = ab_evaluate(board, claim_draw=claim_draw)
            if abs(cp) >= AB_MATE_SCORE - 1000:
                return 1.0 if cp > 0 else -1.0
            return math.tanh(cp / 600.0)
        return evaluator

    raise ValueError(f"Unknown eval mode: {mode}")


def get_eval_mode_name(mode: EvalMode) -> str:
    """Human-readable name for the eval mode."""
    return {
        EvalMode.AB_CLASSIC: "Alpha-Beta Classic",
        EvalMode.MCTS_PE: "MCTS PeSTO/SEE",
        EvalMode.HYBRID: "Hybrid (AB eval + MCTS scaling)",
    }[mode]