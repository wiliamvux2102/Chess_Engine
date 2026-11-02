"""Unified evaluation registry for Alpha-Beta and MCTS.

Allows switching between different evaluation functions:
- 'ab_classic': Original Alpha-Beta evaluation (simple PST + material)
- 'mcts_pe': MCTS PeSTO-style evaluation (tapered eval, SEE, tapered PSQT)
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


def create_evaluator(mode: EvalMode, claim_draw: bool = True) -> Callable[[chess.Board, int], float]:
    """Return an evaluation function matching the specified mode (in centipawns).

    Used for move ordering in root and general position scoring.
    """
    if mode == EvalMode.AB_CLASSIC:
        def evaluator(board: chess.Board, ply: int = 0) -> float:
            return float(ab_evaluate(board, ply=ply, claim_draw=claim_draw))
        return evaluator

    elif mode == EvalMode.MCTS_PE:
        def evaluator(board: chess.Board, ply: int = 0) -> float:
            return mcts_position_score(board)
        return evaluator

    raise ValueError(f"Unknown eval mode: {mode}")


def create_ab_evaluator(mode: EvalMode, claim_draw: bool = True) -> Callable[[chess.Board, int], float]:
    """Return an evaluation function for Alpha-Beta returning centipawn scores."""
    if mode == EvalMode.AB_CLASSIC:
        def evaluator(board: chess.Board, ply: int = 0) -> float:
            return float(ab_evaluate(board, ply=ply, claim_draw=claim_draw))
        return evaluator

    elif mode == EvalMode.MCTS_PE:
        def evaluator(board: chess.Board, ply: int = 0) -> float:
            outcome = board.outcome(claim_draw=claim_draw)
            if outcome is not None:
                if outcome.winner is None:
                    return 0.0
                return float(AB_MATE_SCORE - ply) if outcome.winner == chess.WHITE else float(-AB_MATE_SCORE + ply)
            return mcts_position_score(board)
        return evaluator

    raise ValueError(f"Unknown eval mode: {mode}")


def create_cutoff_evaluator(mode: EvalMode, claim_draw: bool = True) -> Callable[[chess.Board], float]:
    """Return a cutoff evaluator for MCTS rollout termination in [-0.98, +0.98]."""
    if mode == EvalMode.AB_CLASSIC:
        def evaluator(board: chess.Board) -> float:
            outcome = board.outcome(claim_draw=claim_draw)
            if outcome is not None:
                return 0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
            cp = ab_evaluate(board, claim_draw=claim_draw)
            if abs(cp) >= AB_MATE_SCORE - 1000:
                return 0.98 if cp > 0 else -0.98
            return 0.98 * math.tanh(cp / 600.0)
        return evaluator

    elif mode == EvalMode.MCTS_PE:
        return lambda board: mcts_cutoff_value(board)

    raise ValueError(f"Unknown eval mode: {mode}")


def get_eval_mode_name(mode: EvalMode) -> str:
    """Human-readable name for the eval mode."""
    return {
        EvalMode.AB_CLASSIC: "Alpha-Beta Classic",
        EvalMode.MCTS_PE: "MCTS PeSTO/SEE",
    }[mode]