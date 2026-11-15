"""Chess.com-style Evaluation Bar computation and Canvas rendering."""

from __future__ import annotations

import math
import tkinter as tk

import chess

from algorithms.eval_registry import EvalMode
from algorithms.evaluation import MATE_SCORE, evaluate
from algorithms.mcts_rollout import position_score as mcts_position_score
from gui.theme import EVAL_BLACK_BG, EVAL_FONT, EVAL_WHITE_BG, EVAL_WIDTH, MARGIN, SQ_SIZE


def compute_eval_bar_state(
    board: chess.Board,
    *,
    claim_draw: bool = True,
    eval_mode: str | EvalMode = "ab_classic",
) -> tuple[float, str, float | int]:
    """Return (white_ratio in [0.0, 1.0], display_label, raw_centipawn_score)."""
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is not None:
        if outcome.winner == chess.WHITE:
            return 1.0, "1-0", MATE_SCORE
        if outcome.winner == chess.BLACK:
            return 0.0, "0-1", -MATE_SCORE
        return 0.5, "0.0", 0

    mode_str = eval_mode.value if isinstance(eval_mode, EvalMode) else str(eval_mode).lower()
    if mode_str == "mcts_pe":
        cp = mcts_position_score(board)
    else:
        cp = evaluate(board, claim_draw=False)

    if abs(cp) >= MATE_SCORE - 1000:
        return (1.0, "M", cp) if cp > 0 else (0.0, "-M", cp)

    # Logistic centipawn squash (divisor 400)
    ratio = 1.0 / (1.0 + math.exp(-cp / 400.0))
    white_ratio = max(0.05, min(0.95, ratio))
    pawns = cp / 100.0
    label = f"{pawns:+.1f}" if abs(pawns) >= 0.05 else "0.0"
    return white_ratio, label, cp


def render_eval_bar(
    canvas: tk.Canvas,
    board: chess.Board,
    *,
    flipped: bool = False,
    show_eval: bool = True,
    eval_mode: str = "ab_classic",
    claim_draw: bool = True,
    eval_text_var: tk.StringVar | None = None,
) -> None:
    """Draw the vertical evaluation bar on the canvas."""
    white_ratio, label, cp = compute_eval_bar_state(
        board, claim_draw=claim_draw, eval_mode=eval_mode
    )
    if eval_text_var is not None:
        side_adv = (
            "Equal"
            if cp == 0
            else ("White advantage" if cp > 0 else "Black advantage")
        )
        eval_text_var.set(f"Eval ({eval_mode}): {label} ({side_adv})")

    if not show_eval:
        return

    canvas.delete("all")
    top_y = MARGIN
    bar_h = SQ_SIZE * 8
    bot_y = top_y + bar_h
    w = EVAL_WIDTH

    # Determine top vs bottom colors based on board orientation
    if not flipped:
        # White at bottom, Black at top
        top_ratio = 1.0 - white_ratio
        top_color, bot_color = EVAL_BLACK_BG, EVAL_WHITE_BG
        white_on_bottom = True
    else:
        # Black at bottom, White at top
        top_ratio = white_ratio
        top_color, bot_color = EVAL_WHITE_BG, EVAL_BLACK_BG
        white_on_bottom = False

    split_y = top_y + int(round(bar_h * top_ratio))

    canvas.create_rectangle(2, top_y, w - 2, split_y, fill=top_color, outline="")
    canvas.create_rectangle(2, split_y, w - 2, bot_y, fill=bot_color, outline="")
    canvas.create_rectangle(2, top_y, w - 2, bot_y, outline="#555555", width=1)

    # Midline tick mark at 50%
    mid_y = top_y + bar_h // 2
    canvas.create_line(2, mid_y, w - 2, mid_y, fill="#888888", width=1)

    # Place score text in the advantaged side's end (like Chess.com)
    white_winning = cp >= 0
    place_at_bottom = (white_winning and white_on_bottom) or (
        not white_winning and not white_on_bottom
    )
    if place_at_bottom:
        text_y = bot_y - 11
        text_fg = "#262421" if bot_color == EVAL_WHITE_BG else "#f0f0f0"
    else:
        text_y = top_y + 11
        text_fg = "#262421" if top_color == EVAL_WHITE_BG else "#f0f0f0"

    canvas.create_text(
        w // 2,
        text_y,
        text=label,
        fill=text_fg,
        font=EVAL_FONT,
        tags="eval_label",
    )
