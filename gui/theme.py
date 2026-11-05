"""GUI styling tokens, Unicode chess glyphs, and ttk dark theme configuration."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import chess

# Both White and Black use solid-filled Unicode chess glyphs (U+265A..U+265F)
# so White pieces are completely solid inside rather than hollow wireframe outlines.
UNICODE_PIECES: dict[tuple[chess.Color, chess.PieceType], str] = {
    (chess.WHITE, chess.KING): "♚",
    (chess.WHITE, chess.QUEEN): "♛",
    (chess.WHITE, chess.ROOK): "♜",
    (chess.WHITE, chess.BISHOP): "♝",
    (chess.WHITE, chess.KNIGHT): "♞",
    (chess.WHITE, chess.PAWN): "♟",
    (chess.BLACK, chess.KING): "♚",
    (chess.BLACK, chess.QUEEN): "♛",
    (chess.BLACK, chess.ROOK): "♜",
    (chess.BLACK, chess.BISHOP): "♝",
    (chess.BLACK, chess.KNIGHT): "♞",
    (chess.BLACK, chess.PAWN): "♟",
}

PROMOTION_PIECES: tuple[tuple[str, chess.PieceType], ...] = (
    ("Queen", chess.QUEEN),
    ("Knight", chess.KNIGHT),
    ("Rook", chess.ROOK),
    ("Bishop", chess.BISHOP),
)

# Typography
PIECE_FONT: tuple[str, int] = ("Segoe UI Symbol", 34)
PROMO_LABEL_FONT: tuple[str, int, str] = ("Segoe UI", 8, "bold")
COORD_FONT: tuple[str, int, str] = ("Segoe UI", 9, "bold")
EVAL_FONT: tuple[str, int, str] = ("Segoe UI", 8, "bold")

# Board & Eval Bar Dimensions
SQ_SIZE = 68
MARGIN = 26
BOARD_PX = SQ_SIZE * 8 + MARGIN * 2
EVAL_WIDTH = 40

# Board Square Colors
LIGHT_SQ = "#ebecd0"
DARK_SQ = "#739552"
SELECTED_SQ = "#bbcb2b"
LAST_MOVE_LIGHT = "#f5f682"
LAST_MOVE_DARK = "#b9ca43"
CHECK_SQ = "#e53935"

# Evaluation Bar Colors
EVAL_WHITE_BG = "#f0f0f0"
EVAL_BLACK_BG = "#262421"

# Application Dark Theme Palette
BG_MAIN = "#16171a"
BG_CARD = "#23252a"
BG_SUBCARD = "#2c2f36"
BG_INPUT = "#353841"
BG_HOVER = "#444854"
ACCENT = "#81b64c"
ACCENT_HOVER = "#95c95e"
FG_MAIN = "#f3f4f6"
FG_MUTED = "#9ca3af"
BORDER = "#383b44"


def setup_theme(root: tk.Tk | tk.Toplevel) -> None:
    """Configure modern dark theme styles for all ttk widgets."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("TFrame", background=BG_MAIN)
    style.configure("Card.TFrame", background=BG_CARD)
    style.configure("SubCard.TFrame", background=BG_SUBCARD)
    style.configure(
        "TLabelframe",
        background=BG_CARD,
        bordercolor=BORDER,
        darkcolor=BORDER,
        lightcolor=BORDER,
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label",
        background=BG_CARD,
        foreground=ACCENT,
        font=("Segoe UI", 9, "bold"),
    )
    style.configure(
        "TLabel",
        background=BG_CARD,
        foreground=FG_MAIN,
        font=("Segoe UI", 9),
    )
    style.configure(
        "SubCard.TLabel",
        background=BG_SUBCARD,
        foreground=FG_MAIN,
        font=("Segoe UI", 9),
    )
    style.configure(
        "SubMuted.TLabel",
        background=BG_SUBCARD,
        foreground=FG_MUTED,
        font=("Segoe UI", 8, "bold"),
    )
    style.configure(
        "Muted.TLabel",
        background=BG_CARD,
        foreground=FG_MUTED,
        font=("Segoe UI", 9),
    )
    style.configure(
        "TButton",
        background=BG_INPUT,
        foreground=FG_MAIN,
        bordercolor=BORDER,
        focuscolor=BG_INPUT,
        font=("Segoe UI", 9, "bold"),
        padding=(9, 6),
    )
    style.map(
        "TButton",
        background=[("active", BG_HOVER), ("pressed", "#4e5260")],
        foreground=[("active", "#ffffff")],
    )
    style.configure(
        "Accent.TButton",
        background=ACCENT,
        foreground="#16171a",
        bordercolor="#6a9a3b",
        focuscolor=ACCENT,
        font=("Segoe UI", 9, "bold"),
        padding=(10, 6),
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_HOVER), ("pressed", "#6a9a3b")],
        foreground=[("active", "#111214")],
    )
    style.configure(
        "TCheckbutton",
        background=BG_CARD,
        foreground=FG_MAIN,
        font=("Segoe UI", 9),
    )
    style.map(
        "TCheckbutton",
        background=[("active", BG_CARD)],
        foreground=[("active", "#ffffff")],
    )
    style.configure(
        "TCombobox",
        fieldbackground=BG_INPUT,
        background=BG_INPUT,
        foreground=FG_MAIN,
        arrowcolor=FG_MAIN,
        bordercolor=BORDER,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", BG_INPUT)],
        selectbackground=[("readonly", BG_INPUT)],
        selectforeground=[("readonly", FG_MAIN)],
        foreground=[("readonly", FG_MAIN)],
    )
    style.configure(
        "TSpinbox",
        fieldbackground=BG_INPUT,
        background=BG_INPUT,
        foreground=FG_MAIN,
        arrowcolor=FG_MAIN,
        bordercolor=BORDER,
    )
    style.configure(
        "TEntry",
        fieldbackground=BG_INPUT,
        foreground=FG_MAIN,
        insertcolor=FG_MAIN,
        bordercolor=BORDER,
    )
