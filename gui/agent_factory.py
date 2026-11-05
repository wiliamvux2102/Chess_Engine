"""Agent factory and parameter parsing for Alpha-Beta, MCTS, and Human players."""

from __future__ import annotations

import math

from algorithms.alpha_beta import AlphaBetaAI
from algorithms.eval_registry import EvalMode
from algorithms.mcts import MCTS
from chess_system.ai_interface import ChessAI


def parse_optional_int(value: str | int | None, field_name: str) -> int | None:
    """Validate and parse a positive integer or return None for empty input."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return parsed


def parse_optional_float(value: str | float | None, field_name: str) -> float | None:
    """Validate and parse a positive float or return None for empty input."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def build_agent(
    kind: str,
    *,
    depth: str | int | None = 3,
    iterations: str | int | None = 100,
    time_limit: str | float | None = None,
    rollout_max_plies: int | None = 24,
    seed: int | None = None,
    claim_draw: bool = True,
    eval_mode: str | None = None,
) -> ChessAI | None:
    """Create a ChessAI instance from user/CLI parameters, or None for Human."""
    normalized = kind.strip().lower().replace("-", "_")
    if normalized == "human":
        return None
    if normalized in ("alpha_beta", "alphabeta"):
        parsed_depth = parse_optional_int(depth, "Alpha-Beta depth")
        if parsed_depth is None:
            raise ValueError("Alpha-Beta depth is required")
        ab_kwargs: dict[str, object] = {
            "depth": parsed_depth,
            "claim_draw": claim_draw,
        }
        if eval_mode is not None:
            mode_map = {
                "ab_classic": EvalMode.AB_CLASSIC,
                "mcts_pe": EvalMode.MCTS_PE,
            }
            if eval_mode.lower() in mode_map:
                ab_kwargs["eval_mode"] = mode_map[eval_mode.lower()]
            else:
                raise ValueError(f"Unknown eval_mode: {eval_mode}. Valid: {list(mode_map.keys())}")
        return AlphaBetaAI(**ab_kwargs)
    if normalized == "mcts":
        parsed_iter = parse_optional_int(iterations, "MCTS simulations")
        if time_limit is not None and str(time_limit).strip():
            parse_optional_float(time_limit, "MCTS time budget")
        if parsed_iter is None:
            raise ValueError("MCTS requires simulations")
        mcts_kwargs: dict[str, object] = {
            "iterations": parsed_iter,
            "seed": seed,
            "claim_draw": claim_draw,
        }
        if rollout_max_plies is not None:
            mcts_kwargs["rollout_max_plies"] = rollout_max_plies
        if eval_mode is not None:
            mode_map = {
                "ab_classic": EvalMode.AB_CLASSIC,
                "mcts_pe": EvalMode.MCTS_PE,
            }
            if eval_mode.lower() in mode_map:
                mcts_kwargs["eval_mode"] = mode_map[eval_mode.lower()]
            else:
                raise ValueError(f"Unknown eval_mode: {eval_mode}. Valid: {list(mode_map.keys())}")
        return MCTS(**mcts_kwargs)
    raise ValueError(f"Unknown player type: {kind!r}")


def format_search_stats(stats: dict[str, int] | None, elapsed: float) -> str:
    """Format timing and algorithm search statistics into a readable string."""
    if not stats:
        return f"{elapsed:.3f}s"
    parts = [f"{elapsed:.3f}s"]
    for key in ("depth", "nodes", "cutoffs", "iterations", "rollouts"):
        if stats.get(key) is not None:
            parts.append(f"{key}={stats[key]}")
    return " | ".join(parts)
