"""Tkinter 8x8 Chess GUI with Chess.com-style Eval Bar, Human vs AI, and Live AI vs AI."""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from time import perf_counter
from tkinter import messagebox, ttk
from typing import Callable

import chess

from algorithms.alpha_beta import AlphaBetaAI
from algorithms.eval_registry import EvalMode
from algorithms.evaluation import MATE_SCORE, evaluate
from algorithms.mcts import MCTS
from algorithms.mcts_rollout import position_score as mcts_position_score
from chess_system.ai_interface import ChessAI
from chess_system.game_controller import GameController

# Both White and Black use the solid-filled Unicode chess glyphs (U+265A..U+265F)
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

PIECE_FONT: tuple[str, int] = ("Segoe UI Symbol", 34)
PROMO_LABEL_FONT: tuple[str, int, str] = ("Segoe UI", 8, "bold")
COORD_FONT: tuple[str, int, str] = ("Segoe UI", 9, "bold")
EVAL_FONT: tuple[str, int, str] = ("Segoe UI", 8, "bold")


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

    # ponytail: logistic centipawn squash (divisor 400); upgrade to WDL lookup if NNUE added
    ratio = 1.0 / (1.0 + math.exp(-cp / 400.0))
    white_ratio = max(0.05, min(0.95, ratio))
    pawns = cp / 100.0
    label = f"{pawns:+.1f}" if abs(pawns) >= 0.05 else "0.0"
    return white_ratio, label, cp


def parse_optional_int(value: str | int | None, field_name: str) -> int | None:
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
    if not stats:
        return f"{elapsed:.3f}s"
    parts = [f"{elapsed:.3f}s"]
    for key in ("depth", "nodes", "cutoffs", "iterations", "rollouts"):
        if stats.get(key) is not None:
            parts.append(f"{key}={stats[key]}")
    return " | ".join(parts)


class ChessGUI:
    """Interactive 8x8 Chess GUI supporting Human vs AI and Live AI vs AI."""

    SQ_SIZE = 68
    MARGIN = 26
    BOARD_PX = SQ_SIZE * 8 + MARGIN * 2
    EVAL_WIDTH = 40

    LIGHT_SQ = "#ebecd0"
    DARK_SQ = "#739552"
    SELECTED_SQ = "#bbcb2b"
    LAST_MOVE_LIGHT = "#f5f682"
    LAST_MOVE_DARK = "#b9ca43"
    CHECK_SQ = "#e53935"
    EVAL_WHITE_BG = "#f0f0f0"
    EVAL_BLACK_BG = "#262421"

    def __init__(
        self,
        root: tk.Tk | None = None,
        *,
        fen: str = chess.STARTING_FEN,
        white_kind: str = "human",
        black_kind: str = "alpha_beta",
        white_depth: int = 3,
        black_depth: int = 3,
        white_iterations: int = 100,
        black_iterations: int = 100,
        white_time_limit: float | None = None,
        black_time_limit: float | None = None,
        white_eval_mode: str = "ab_classic",
        black_eval_mode: str = "mcts_pe",
        eval_bar_mode: str = "ab_classic",
        move_delay_ms: int = 300,
        show_eval_bar: bool = True,
        auto_start: bool = True,
        promotion_chooser: Callable[[chess.Color], chess.PieceType] | None = None,
        error_handler: Callable[[str, str], None] | None = None,
    ):
        self.owns_root = root is None
        self.root = root if root is not None else tk.Tk()
        self.root.title("Chess Engine — Alpha-Beta & MCTS")
        self.root.configure(bg="#16171a")

        self.promotion_chooser = promotion_chooser
        self.error_handler = error_handler or messagebox.showerror
        self.last_error: str | None = None

        self.controller = GameController(fen, claim_draw=True)
        self.agents: dict[chess.Color, ChessAI | None] = {
            chess.WHITE: None,
            chess.BLACK: None,
        }
        self.player_names: dict[chess.Color, str] = {
            chess.WHITE: "Human",
            chess.BLACK: "Alpha-Beta",
        }

        self.selected_square: chess.Square | None = None
        self.legal_targets: dict[chess.Square, list[chess.Move]] = {}
        self.pending_promotion: tuple[chess.Square, chess.Square, list[chess.Move]] | None = None
        self.hover_promo_sq: chess.Square | None = None
        self.last_move: chess.Move | None = None
        self.flipped: bool = False
        self.paused: bool = False
        self.ai_thinking: bool = False
        self.ai_worker_thread: threading.Thread | None = None
        self.search_token: int = 0
        self.after_id: str | None = None
        self.poll_after_id: str | None = None
        self.ai_queue: queue.Queue[tuple[int, chess.Move | None, float, dict, Exception | None]] = (
            queue.Queue()
        )
        self.move_log: list[str] = []
        self.side_controls: dict[str, dict[str, tk.Widget]] = {}

        # Tkinter Variables
        self.white_kind_var = tk.StringVar(value=white_kind)
        self.black_kind_var = tk.StringVar(value=black_kind)
        self.white_depth_var = tk.StringVar(value=str(white_depth))
        self.black_depth_var = tk.StringVar(value=str(black_depth))
        self.white_iter_var = tk.StringVar(value=str(white_iterations))
        self.black_iter_var = tk.StringVar(value=str(black_iterations))
        self.white_time_var = tk.StringVar(
            value="" if white_time_limit is None else str(white_time_limit)
        )
        self.black_time_var = tk.StringVar(
            value="" if black_time_limit is None else str(black_time_limit)
        )
        self.white_eval_mode_var = tk.StringVar(value=white_eval_mode)
        self.black_eval_mode_var = tk.StringVar(value=black_eval_mode)
        self.eval_bar_mode_var = tk.StringVar(value=eval_bar_mode)
        self.delay_var = tk.StringVar(value=str(max(0, int(move_delay_ms))))
        self.fen_var = tk.StringVar(value=fen)
        self.show_eval_var = tk.BooleanVar(value=show_eval_bar)

        self.status_var = tk.StringVar(value="Ready")
        self.eval_text_var = tk.StringVar(value="Eval: 0.0")
        self.stats_var = tk.StringVar(value="Stats: -")
        self.pause_btn_text = tk.StringVar(value="Pause")

        self._setup_theme()
        self._build_ui()
        self.toggle_eval_bar(show_eval_bar)
        self.new_game(auto_start=auto_start)
        self.poll_after_id = self.root.after(25, self._poll_ai_queue)
        if self.owns_root:
            self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _setup_theme(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        bg_main = "#16171a"
        bg_card = "#23252a"
        bg_subcard = "#2c2f36"
        bg_input = "#353841"
        bg_hover = "#444854"
        accent = "#81b64c"
        accent_hover = "#95c95e"
        fg_main = "#f3f4f6"
        fg_muted = "#9ca3af"
        border = "#383b44"

        style.configure("TFrame", background=bg_main)
        style.configure("Card.TFrame", background=bg_card)
        style.configure("SubCard.TFrame", background=bg_subcard)
        style.configure(
            "TLabelframe",
            background=bg_card,
            bordercolor=border,
            darkcolor=border,
            lightcolor=border,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background=bg_card,
            foreground=accent,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "TLabel",
            background=bg_card,
            foreground=fg_main,
            font=("Segoe UI", 9),
        )
        style.configure(
            "SubCard.TLabel",
            background=bg_subcard,
            foreground=fg_main,
            font=("Segoe UI", 9),
        )
        style.configure(
            "SubMuted.TLabel",
            background=bg_subcard,
            foreground=fg_muted,
            font=("Segoe UI", 8, "bold"),
        )
        style.configure(
            "Muted.TLabel",
            background=bg_card,
            foreground=fg_muted,
            font=("Segoe UI", 9),
        )
        style.configure(
            "TButton",
            background=bg_input,
            foreground=fg_main,
            bordercolor=border,
            focuscolor=bg_input,
            font=("Segoe UI", 9, "bold"),
            padding=(9, 6),
        )
        style.map(
            "TButton",
            background=[("active", bg_hover), ("pressed", "#4e5260")],
            foreground=[("active", "#ffffff")],
        )
        style.configure(
            "Accent.TButton",
            background=accent,
            foreground="#16171a",
            bordercolor="#6a9a3b",
            focuscolor=accent,
            font=("Segoe UI", 9, "bold"),
            padding=(10, 6),
        )
        style.map(
            "Accent.TButton",
            background=[("active", accent_hover), ("pressed", "#6a9a3b")],
            foreground=[("active", "#111214")],
        )
        style.configure(
            "TCheckbutton",
            background=bg_card,
            foreground=fg_main,
            font=("Segoe UI", 9),
        )
        style.map(
            "TCheckbutton",
            background=[("active", bg_card)],
            foreground=[("active", "#ffffff")],
        )
        style.configure(
            "TCombobox",
            fieldbackground=bg_input,
            background=bg_input,
            foreground=fg_main,
            arrowcolor=fg_main,
            bordercolor=border,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", bg_input)],
            selectbackground=[("readonly", bg_input)],
            selectforeground=[("readonly", fg_main)],
            foreground=[("readonly", fg_main)],
        )
        style.configure(
            "TSpinbox",
            fieldbackground=bg_input,
            background=bg_input,
            foreground=fg_main,
            arrowcolor=fg_main,
            bordercolor=border,
        )
        style.configure(
            "TEntry",
            fieldbackground=bg_input,
            foreground=fg_main,
            insertcolor=fg_main,
            bordercolor=border,
        )

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=14)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Left Area: Eval Bar (Col 0) + 8x8 Board Canvas (Col 1)
        self.board_container = ttk.Frame(main_frame)
        self.board_container.grid(row=0, column=0, padx=(0, 14), sticky="n")

        self.eval_canvas = tk.Canvas(
            self.board_container,
            width=self.EVAL_WIDTH,
            height=self.BOARD_PX,
            bg="#16171a",
            highlightthickness=1,
            highlightbackground="#3f4147",
        )
        self.eval_canvas.grid(row=0, column=0, padx=(0, 8))

        self.board_canvas = tk.Canvas(
            self.board_container,
            width=self.BOARD_PX,
            height=self.BOARD_PX,
            bg="#16171a",
            highlightthickness=1,
            highlightbackground="#3f4147",
        )
        self.board_canvas.grid(row=0, column=1)
        self.board_canvas.bind("<Button-1>", self._on_canvas_click)
        self.board_canvas.bind("<Motion>", self._on_canvas_motion)

        # Right Area: Controls, Config, Status, Move Log
        ctrl = ttk.Frame(main_frame)
        ctrl.grid(row=0, column=1, sticky="nsew")

        # 1. Quick Mode Presets
        preset_box = ttk.LabelFrame(ctrl, text="Quick Mode Presets", padding=8)
        preset_box.pack(fill="x", pady=(0, 8))
        ttk.Button(
            preset_box,
            text="Human vs AI",
            command=lambda: self.apply_preset("human", "alpha_beta", white_eval="ab_classic", black_eval="ab_classic"),
        ).grid(row=0, column=0, padx=3, pady=2, sticky="ew")
        ttk.Button(
            preset_box,
            text="AI vs Human",
            command=lambda: self.apply_preset("mcts", "human", white_eval="mcts_pe", black_eval="ab_classic"),
        ).grid(row=0, column=1, padx=3, pady=2, sticky="ew")
        ttk.Button(
            preset_box,
            text="Live AI vs AI",
            command=lambda: self.apply_preset("alpha_beta", "mcts", white_eval="mcts_pe", black_eval="mcts_pe"),
        ).grid(row=0, column=2, padx=3, pady=2, sticky="ew")
        ttk.Button(
            preset_box,
            text="Human vs Human",
            command=lambda: self.apply_preset("human", "human"),
        ).grid(row=0, column=3, padx=3, pady=2, sticky="ew")
        for col in range(4):
            preset_box.columnconfigure(col, weight=1)

        # 2. Player & AI Configuration (Dynamic per selected algorithm)
        cfg_box = ttk.LabelFrame(ctrl, text="Players & Search Parameters", padding=8)
        cfg_box.pack(fill="x", pady=(0, 8))

        self._build_side_row(
            cfg_box,
            side_key="white",
            label="White",
            kind_var=self.white_kind_var,
            eval_var=self.white_eval_mode_var,
            depth_var=self.white_depth_var,
            iter_var=self.white_iter_var,
        )
        self._build_side_row(
            cfg_box,
            side_key="black",
            label="Black",
            kind_var=self.black_kind_var,
            eval_var=self.black_eval_mode_var,
            depth_var=self.black_depth_var,
            iter_var=self.black_iter_var,
        )

        # FEN Row
        fen_frame = ttk.Frame(cfg_box, style="Card.TFrame")
        fen_frame.pack(fill="x", pady=(6, 0))
        ttk.Label(fen_frame, text="FEN:", style="Muted.TLabel").pack(side="left")
        ttk.Entry(fen_frame, textvariable=self.fen_var, width=32).pack(
            side="left", padx=6, fill="x", expand=True
        )
        ttk.Button(
            fen_frame,
            text="Reset FEN",
            width=10,
            command=lambda: self.fen_var.set(chess.STARTING_FEN),
        ).pack(side="left")

        # 3. Live Match Controls & View Options
        action_box = ttk.LabelFrame(ctrl, text="Match & View Controls", padding=8)
        action_box.pack(fill="x", pady=(0, 8))

        btn_row = ttk.Frame(action_box, style="Card.TFrame")
        btn_row.pack(fill="x", pady=2)
        ttk.Button(
            btn_row,
            text="New Game",
            style="Accent.TButton",
            command=self.new_game,
        ).pack(side="left", padx=2, expand=True, fill="x")
        self.pause_btn = ttk.Button(
            btn_row, textvariable=self.pause_btn_text, command=self.toggle_pause
        )
        self.pause_btn.pack(side="left", padx=2, expand=True, fill="x")
        ttk.Button(btn_row, text="Step (1 Move)", command=self.step_ai_move).pack(
            side="left", padx=2, expand=True, fill="x"
        )
        ttk.Button(btn_row, text="Undo", command=self.undo_move).pack(
            side="left", padx=2, expand=True, fill="x"
        )
        ttk.Button(btn_row, text="Flip Board", command=self.flip_board).pack(
            side="left", padx=2, expand=True, fill="x"
        )

        opt_row = ttk.Frame(action_box, style="Card.TFrame")
        opt_row.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            opt_row,
            text="Show Eval Bar",
            variable=self.show_eval_var,
            command=lambda: self.toggle_eval_bar(self.show_eval_var.get()),
        ).pack(side="left", padx=(2, 6))

        ttk.Label(opt_row, text="Bar Eval:", style="Muted.TLabel").pack(side="left", padx=(4, 2))
        bar_combo = ttk.Combobox(
            opt_row,
            textvariable=self.eval_bar_mode_var,
            values=("ab_classic", "mcts_pe"),
            state="readonly",
            width=10,
        )
        bar_combo.pack(side="left", padx=(0, 8))
        bar_combo.bind("<<ComboboxSelected>>", lambda e: self._draw_eval_bar())

        ttk.Label(opt_row, text="AI Delay (ms):").pack(side="left")
        ttk.Spinbox(
            opt_row,
            from_=0,
            to=3000,
            increment=100,
            textvariable=self.delay_var,
            width=6,
        ).pack(side="left", padx=4)

        # 4. Status & Search Metrics
        status_box = ttk.LabelFrame(ctrl, text="Game Status & AI Telemetry", padding=8)
        status_box.pack(fill="x", pady=(0, 8))
        ttk.Label(
            status_box,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
            foreground="#ffffff",
            wraplength=370,
        ).pack(anchor="w")
        ttk.Label(
            status_box,
            textvariable=self.eval_text_var,
            font=("Segoe UI", 9, "bold"),
            foreground="#81b64c",
        ).pack(anchor="w", pady=(3, 0))
        ttk.Label(
            status_box,
            textvariable=self.stats_var,
            style="Muted.TLabel",
            font=("Consolas", 9),
            wraplength=370,
        ).pack(anchor="w", pady=(3, 0))

        # 5. Live Move History
        hist_box = ttk.LabelFrame(ctrl, text="Move History", padding=8)
        hist_box.pack(fill="both", expand=True)
        self.history_list = tk.Listbox(
            hist_box,
            font=("Consolas", 9),
            height=12,
            bg="#1e1f22",
            fg="#e3e5e8",
            selectbackground="#404249",
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#3f4147",
            relief="flat",
            borderwidth=4,
            activestyle="none",
        )
        scrollbar = ttk.Scrollbar(
            hist_box, orient="vertical", command=self.history_list.yview
        )
        self.history_list.configure(yscrollcommand=scrollbar.set)
        self.history_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_side_row(
        self,
        parent: ttk.Widget,
        *,
        side_key: str,
        label: str,
        kind_var: tk.StringVar,
        eval_var: tk.StringVar,
        depth_var: tk.StringVar,
        iter_var: tk.StringVar,
    ) -> None:
        card = ttk.Frame(parent, style="SubCard.TFrame", padding=(8, 6))
        card.pack(fill="x", pady=(0, 6))

        # Explicit player piece badge (White = pure white token, Black = dark token)
        badge = tk.Canvas(
            card,
            width=16,
            height=16,
            bg="#2c2f36",
            highlightthickness=0,
        )
        badge.pack(side="left", padx=(0, 6))
        if side_key == "white":
            badge.create_oval(1, 1, 15, 15, fill="#ffffff", outline="#b5bac1", width=1)
        else:
            badge.create_oval(1, 1, 15, 15, fill="#121316", outline="#6b7280", width=1)

        ttk.Label(
            card, text=label, style="SubCard.TLabel", font=("Segoe UI", 9, "bold"), width=7
        ).pack(side="left")

        combo = ttk.Combobox(
            card,
            textvariable=kind_var,
            values=("human", "alpha_beta", "mcts"),
            state="readonly",
            width=11,
        )
        combo.pack(side="left", padx=(4, 10))

        params_frame = ttk.Frame(card, style="SubCard.TFrame")
        params_frame.pack(side="left", fill="x", expand=True)

        human_lbl = ttk.Label(
            params_frame, text="Manual Board Control", style="SubMuted.TLabel"
        )

        eval_frame = ttk.Frame(params_frame, style="SubCard.TFrame")
        ttk.Label(eval_frame, text="Eval:", style="SubMuted.TLabel").pack(side="left", padx=(0, 3))
        ttk.Combobox(
            eval_frame,
            textvariable=eval_var,
            values=("ab_classic", "mcts_pe"),
            state="readonly",
            width=10,
        ).pack(side="left", padx=(0, 8))

        depth_frame = ttk.Frame(params_frame, style="SubCard.TFrame")
        ttk.Label(depth_frame, text="Depth:", style="SubMuted.TLabel").pack(side="left", padx=(0, 3))
        ttk.Spinbox(depth_frame, from_=1, to=8, textvariable=depth_var, width=5).pack(side="left")

        iter_frame = ttk.Frame(params_frame, style="SubCard.TFrame")
        ttk.Label(iter_frame, text="Sims:", style="SubMuted.TLabel").pack(side="left", padx=(0, 3))
        ttk.Spinbox(
            iter_frame, from_=1, to=20000, increment=50, textvariable=iter_var, width=7
        ).pack(side="left")

        self.side_controls[side_key] = {
            "human_lbl": human_lbl,
            "eval_frame": eval_frame,
            "depth_frame": depth_frame,
            "iter_frame": iter_frame,
        }
        kind_var.trace_add("write", lambda *_: self._sync_side_controls(side_key, kind_var.get()))
        self._sync_side_controls(side_key, kind_var.get())

    def _sync_side_controls(self, side_key: str, kind: str) -> None:
        widgets = self.side_controls.get(side_key)
        if not widgets:
            return
        for w in widgets.values():
            w.pack_forget()

        norm = kind.strip().lower().replace("-", "_")
        if norm in ("alpha_beta", "alphabeta"):
            widgets["eval_frame"].pack(side="left")
            widgets["depth_frame"].pack(side="left")
        elif norm == "mcts":
            widgets["eval_frame"].pack(side="left")
            widgets["iter_frame"].pack(side="left")
        else:
            widgets["human_lbl"].pack(side="left")

    def _report_error(self, title: str, message: str) -> None:
        self.last_error = message
        self.status_var.set(f"Error: {message}")
        self.error_handler(title, message)

    def toggle_eval_bar(self, show: bool | None = None) -> bool:
        """Show or hide the left-side Chess.com-style evaluation bar."""
        if show is None:
            show = not bool(self.show_eval_var.get())
        self.show_eval_var.set(bool(show))
        if show:
            self.eval_canvas.grid(row=0, column=0, padx=(0, 6))
            self._draw_eval_bar()
        else:
            self.eval_canvas.grid_remove()
        return bool(show)

    def apply_preset(
        self,
        white_kind: str,
        black_kind: str,
        *,
        white_eval: str | None = None,
        black_eval: str | None = None,
        auto_start: bool = True,
    ) -> None:
        self.white_kind_var.set(white_kind)
        self.black_kind_var.set(black_kind)
        if white_eval is not None:
            self.white_eval_mode_var.set(white_eval)
        if black_eval is not None:
            self.black_eval_mode_var.set(black_eval)
        self.new_game(auto_start=auto_start)

    def _parse_delay_ms(self) -> int:
        try:
            return max(0, int(float(self.delay_var.get().strip() or "0")))
        except ValueError:
            return 500

    def _cancel_pending_ai(self) -> None:
        self.search_token += 1
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

        # Abort any in-progress search on active AI agents
        for agent in self.agents.values():
            if agent is not None and hasattr(agent, "abort_search"):
                try:
                    agent.abort_search()
                except Exception:
                    pass

        # Wait for the background worker thread to stop to prevent concurrent execution
        if self.ai_worker_thread is not None and self.ai_worker_thread.is_alive():
            self.ai_worker_thread.join(timeout=0.3)
            if not self.ai_worker_thread.is_alive():
                self.ai_worker_thread = None

        self.ai_thinking = (self.ai_worker_thread is not None and self.ai_worker_thread.is_alive())

    def new_game(self, *, fen: str | None = None, auto_start: bool = True) -> bool:
        """Validate configuration, reset board, and start a fresh game."""
        target_fen = (fen if fen is not None else self.fen_var.get()).strip() or chess.STARTING_FEN
        try:
            new_controller = GameController(target_fen, claim_draw=True)
            white_agent = build_agent(
                self.white_kind_var.get(),
                depth=self.white_depth_var.get(),
                iterations=self.white_iter_var.get(),
                time_limit=self.white_time_var.get(),
                rollout_max_plies=24,
                seed=None,
                claim_draw=new_controller.claim_draw,
                eval_mode=self.white_eval_mode_var.get(),
            )
            black_agent = build_agent(
                self.black_kind_var.get(),
                depth=self.black_depth_var.get(),
                iterations=self.black_iter_var.get(),
                time_limit=self.black_time_var.get(),
                rollout_max_plies=24,
                seed=None,
                claim_draw=new_controller.claim_draw,
                eval_mode=self.black_eval_mode_var.get(),
            )
        except ValueError as exc:
            self._report_error("Invalid Configuration", str(exc))
            return False

        self._cancel_pending_ai()
        self.last_error = None
        self.controller = new_controller
        self.fen_var.set(target_fen)
        self.agents = {chess.WHITE: white_agent, chess.BLACK: black_agent}
        self.player_names = {
            chess.WHITE: self._format_agent_label(self.white_kind_var.get(), self.white_eval_mode_var.get()),
            chess.BLACK: self._format_agent_label(self.black_kind_var.get(), self.black_eval_mode_var.get()),
        }
        self.selected_square = None
        self.legal_targets = {}
        self.pending_promotion = None
        self.hover_promo_sq = None
        self.last_move = None
        self.paused = not auto_start
        self.pause_btn_text.set("Resume" if self.paused else "Pause")
        self.move_log.clear()
        self.history_list.delete(0, tk.END)
        self.stats_var.set("Stats: -")

        self._refresh_ui()
        if auto_start:
            self._maybe_schedule_ai()
        return True

    @staticmethod
    def _format_agent_label(kind: str, eval_mode: str | None = None) -> str:
        norm = kind.strip().lower().replace("-", "_")
        if norm == "human":
            return "Human"
        mode_suffix = f" [{eval_mode}]" if eval_mode else ""
        if norm in ("alpha_beta", "alphabeta"):
            return f"Alpha-Beta{mode_suffix}"
        if norm == "mcts":
            return f"MCTS{mode_suffix}"
        return f"{kind}{mode_suffix}"

    def current_agent(self) -> ChessAI | None:
        return self.agents[self.controller.turn]

    def flip_board(self) -> None:
        self.flipped = not self.flipped
        self._refresh_ui()

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        self.pause_btn_text.set("Resume" if self.paused else "Pause")
        if self.paused:
            self._cancel_pending_ai()
            self._refresh_ui()
        else:
            self._refresh_ui()
            self._maybe_schedule_ai()
        return self.paused

    def step_ai_move(self) -> bool:
        """Let the current AI compute and play exactly 1 move on the board, then pause."""
        if (
            self.controller.is_terminal()
            or self.current_agent() is None
            or self.ai_thinking
            or (self.ai_worker_thread is not None and self.ai_worker_thread.is_alive())
        ):
            return False
        self.paused = True
        self.pause_btn_text.set("Resume")
        self._start_ai_turn()
        return True

    def undo_move(self) -> bool:
        """Undo last ply (or human+AI pair in Human vs AI) and cancel active search."""
        self._cancel_pending_ai()
        board_stack = self.controller.board.move_stack
        if not board_stack:
            return False

        # If Human vs AI and it is currently Human's turn, undo both AI and Human moves if possible
        current_is_human = self.current_agent() is None
        other_is_ai = self.agents[not self.controller.turn] is not None
        plies_to_pop = 2 if (current_is_human and other_is_ai and len(board_stack) >= 2) else 1

        for _ in range(plies_to_pop):
            try:
                self.controller.pop()
            except IndexError:
                break
            if self.move_log:
                self.move_log.pop()
                self.history_list.delete(tk.END)

        current_stack = self.controller.board.move_stack
        self.last_move = current_stack[-1] if current_stack else None
        self.selected_square = None
        self.legal_targets = {}
        self.pending_promotion = None
        self.hover_promo_sq = None

        # If the resulting turn is an AI, pause so it doesn't immediately redo the move
        if self.current_agent() is not None:
            self.paused = True
            self.pause_btn_text.set("Resume")

        self._refresh_ui()
        return True

    # ============================================================
    # Board Coordinate Helpers & Rendering
    # ============================================================

    def square_to_grid(self, square: chess.Square) -> tuple[int, int]:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if self.flipped:
            return 7 - file_idx, rank_idx
        return file_idx, 7 - rank_idx

    def pixel_to_square(self, x: int, y: int) -> chess.Square | None:
        col = (x - self.MARGIN) // self.SQ_SIZE
        row = (y - self.MARGIN) // self.SQ_SIZE
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        file_idx = 7 - col if self.flipped else col
        rank_idx = row if self.flipped else 7 - row
        return chess.square(file_idx, rank_idx)

    def _refresh_ui(self, board: chess.Board | None = None) -> None:
        if board is None:
            board = self.controller.board
        self._draw_board(board)
        self._draw_eval_bar(board)
        self._update_status_text(board)

    def _draw_eval_bar(self, board: chess.Board | None = None) -> None:
        if board is None:
            board = self.controller.board
        bar_mode = getattr(self, "eval_bar_mode_var", None)
        mode_val = bar_mode.get() if bar_mode is not None else "ab_classic"
        white_ratio, label, cp = compute_eval_bar_state(
            board, claim_draw=self.controller.claim_draw, eval_mode=mode_val
        )
        side_adv = (
            "Equal"
            if cp == 0
            else ("White advantage" if cp > 0 else "Black advantage")
        )
        self.eval_text_var.set(f"Eval ({mode_val}): {label} ({side_adv})")

        if not self.show_eval_var.get():
            return

        self.eval_canvas.delete("all")
        top_y = self.MARGIN
        bar_h = self.SQ_SIZE * 8
        bot_y = top_y + bar_h
        w = self.EVAL_WIDTH

        # Determine top vs bottom colors based on board orientation
        if not self.flipped:
            # White at bottom, Black at top
            top_ratio = 1.0 - white_ratio
            top_color, bot_color = self.EVAL_BLACK_BG, self.EVAL_WHITE_BG
            white_on_bottom = True
        else:
            # Black at bottom, White at top
            top_ratio = white_ratio
            top_color, bot_color = self.EVAL_WHITE_BG, self.EVAL_BLACK_BG
            white_on_bottom = False

        split_y = top_y + int(round(bar_h * top_ratio))

        self.eval_canvas.create_rectangle(
            2, top_y, w - 2, split_y, fill=top_color, outline=""
        )
        self.eval_canvas.create_rectangle(
            2, split_y, w - 2, bot_y, fill=bot_color, outline=""
        )
        self.eval_canvas.create_rectangle(
            2, top_y, w - 2, bot_y, outline="#555555", width=1
        )

        # Midline tick mark at 50%
        mid_y = top_y + bar_h // 2
        self.eval_canvas.create_line(2, mid_y, w - 2, mid_y, fill="#888888", width=1)

        # Place score text in the advantaged side's end (like Chess.com)
        white_winning = cp >= 0
        place_at_bottom = (white_winning and white_on_bottom) or (
            not white_winning and not white_on_bottom
        )
        if place_at_bottom:
            text_y = bot_y - 11
            text_fg = "#262421" if bot_color == self.EVAL_WHITE_BG else "#f0f0f0"
        else:
            text_y = top_y + 11
            text_fg = "#262421" if top_color == self.EVAL_WHITE_BG else "#f0f0f0"

        self.eval_canvas.create_text(
            w // 2,
            text_y,
            text=label,
            fill=text_fg,
            font=EVAL_FONT,
            tags="eval_label",
        )

    def _draw_board(self, board: chess.Board | None = None) -> None:
        if board is None:
            board = self.controller.board
        self.board_canvas.delete("all")
        king_in_check_sq = board.king(board.turn) if board.is_check() else None
        last_squares = (
            {self.last_move.from_square, self.last_move.to_square}
            if self.last_move is not None
            else set()
        )

        for square in chess.SQUARES:
            col, row = self.square_to_grid(square)
            x1 = self.MARGIN + col * self.SQ_SIZE
            y1 = self.MARGIN + row * self.SQ_SIZE
            x2 = x1 + self.SQ_SIZE
            y2 = y1 + self.SQ_SIZE

            is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            fill = self.LIGHT_SQ if is_light else self.DARK_SQ
            if square in last_squares:
                fill = self.LAST_MOVE_LIGHT if is_light else self.LAST_MOVE_DARK
            if square == self.selected_square:
                fill = self.SELECTED_SQ
            if square == king_in_check_sq:
                fill = self.CHECK_SQ

            self.board_canvas.create_rectangle(
                x1, y1, x2, y2, fill=fill, outline="", tags=f"sq_{square}"
            )

            # Legal target indicator
            if square in self.legal_targets:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if board.piece_at(square) is not None or (
                    self.selected_square is not None
                    and board.is_en_passant(self.legal_targets[square][0])
                ):
                    self.board_canvas.create_oval(
                        x1 + 4,
                        y1 + 4,
                        x2 - 4,
                        y2 - 4,
                        outline="#3a6b35",
                        width=4,
                        tags="legal_hint",
                    )
                else:
                    r = 9
                    self.board_canvas.create_oval(
                        cx - r,
                        cy - r,
                        cx + r,
                        cy + r,
                        fill="#4a7846",
                        outline="",
                        tags="legal_hint",
                    )

            # Piece glyph (solid silhouette for both White and Black)
            piece = board.piece_at(square)
            if piece is not None:
                glyph = UNICODE_PIECES[(piece.color, piece.piece_type)]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if piece.color == chess.WHITE:
                    # Seal internal slits and create a crisp dark outline around the solid glyph
                    for dx, dy in (
                        (-2, 0),
                        (2, 0),
                        (0, -2),
                        (0, 2),
                        (-1, -1),
                        (1, -1),
                        (-1, 1),
                        (1, 1),
                    ):
                        self.board_canvas.create_text(
                            cx + dx,
                            cy + dy,
                            text=glyph,
                            fill="#18191c",
                            font=PIECE_FONT,
                        )
                    self.board_canvas.create_text(
                        cx,
                        cy,
                        text=glyph,
                        fill="#ffffff",
                        font=PIECE_FONT,
                        tags="piece",
                    )
                else:
                    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                        self.board_canvas.create_text(
                            cx + dx,
                            cy + dy,
                            text=glyph,
                            fill="#d4d4d8",
                            font=PIECE_FONT,
                        )
                    self.board_canvas.create_text(
                        cx,
                        cy,
                        text=glyph,
                        fill="#18191c",
                        font=PIECE_FONT,
                        tags="piece",
                    )

        # Outer border around the 8x8 playing grid
        grid_end = self.MARGIN + 8 * self.SQ_SIZE
        self.board_canvas.create_rectangle(
            self.MARGIN,
            self.MARGIN,
            grid_end,
            grid_end,
            outline="#3f4147",
            width=2,
        )

        # Draw file and rank coordinates
        for idx in range(8):
            file_char = chess.FILE_NAMES[7 - idx if self.flipped else idx]
            rank_char = chess.RANK_NAMES[idx if self.flipped else 7 - idx]
            fx = self.MARGIN + idx * self.SQ_SIZE + self.SQ_SIZE // 2
            ry = self.MARGIN + idx * self.SQ_SIZE + self.SQ_SIZE // 2
            self.board_canvas.create_text(
                fx,
                self.BOARD_PX - self.MARGIN // 2,
                text=file_char,
                fill="#b5bac1",
                font=COORD_FONT,
            )
            self.board_canvas.create_text(
                self.MARGIN // 2,
                ry,
                text=rank_char,
                fill="#b5bac1",
                font=COORD_FONT,
            )

        # Chess.com-style in-board promotion column overlay
        if self.pending_promotion is not None:
            promo_targets = self._get_promotion_targets()
            if promo_targets:
                cols = [c for _, c, _, _, _ in promo_targets]
                rows = [r for _, _, r, _, _ in promo_targets]
                px1 = self.MARGIN + min(cols) * self.SQ_SIZE
                py1 = self.MARGIN + min(rows) * self.SQ_SIZE
                px2 = self.MARGIN + (max(cols) + 1) * self.SQ_SIZE
                py2 = self.MARGIN + (max(rows) + 1) * self.SQ_SIZE

                # Drop shadow + elevated card container
                self.board_canvas.create_rectangle(
                    px1 + 3,
                    py1 + 3,
                    px2 + 3,
                    py2 + 3,
                    fill="#0d0e10",
                    outline="",
                    tags="promo_overlay",
                )
                self.board_canvas.create_rectangle(
                    px1,
                    py1,
                    px2,
                    py2,
                    fill="#f8f9fa",
                    outline="#81b64c",
                    width=3,
                    tags="promo_overlay",
                )

                promo_color = board.turn
                for opt_sq, col, row, pt, label in promo_targets:
                    sx1 = self.MARGIN + col * self.SQ_SIZE
                    sy1 = self.MARGIN + row * self.SQ_SIZE
                    sx2 = sx1 + self.SQ_SIZE
                    sy2 = sy1 + self.SQ_SIZE
                    cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                    hovered = opt_sq == self.hover_promo_sq

                    bg_fill = "#e2f0cb" if hovered else "#f8f9fa"
                    self.board_canvas.create_rectangle(
                        sx1 + 2,
                        sy1 + 2,
                        sx2 - 2,
                        sy2 - 2,
                        fill=bg_fill,
                        outline="#d0d4dc",
                        width=1,
                        tags="promo_overlay",
                    )
                    disc_r = self.SQ_SIZE // 2 - 5
                    disc_fill = "#ffffff" if hovered else "#e9ecef"
                    disc_outline = "#81b64c" if hovered else "#ced4da"
                    self.board_canvas.create_oval(
                        cx - disc_r,
                        cy - disc_r,
                        cx + disc_r,
                        cy + disc_r,
                        fill=disc_fill,
                        outline=disc_outline,
                        width=2 if hovered else 1,
                        tags="promo_overlay",
                    )

                    glyph = UNICODE_PIECES[(promo_color, pt)]
                    if promo_color == chess.WHITE:
                        for dx, dy in (
                            (-2, 0),
                            (2, 0),
                            (0, -2),
                            (0, 2),
                            (-1, -1),
                            (1, -1),
                            (-1, 1),
                            (1, 1),
                        ):
                            self.board_canvas.create_text(
                                cx + dx,
                                cy + dy,
                                text=glyph,
                                fill="#18191c",
                                font=PIECE_FONT,
                                tags="promo_overlay",
                            )
                        self.board_canvas.create_text(
                            cx,
                            cy,
                            text=glyph,
                            fill="#ffffff",
                            font=PIECE_FONT,
                            tags="promo_overlay",
                        )
                    else:
                        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                            self.board_canvas.create_text(
                                cx + dx,
                                cy + dy,
                                text=glyph,
                                fill="#ffffff",
                                font=PIECE_FONT,
                                tags="promo_overlay",
                            )
                        self.board_canvas.create_text(
                            cx,
                            cy,
                            text=glyph,
                            fill="#18191c",
                            font=PIECE_FONT,
                            tags="promo_overlay",
                        )

                    self.board_canvas.create_text(
                        sx2 - 9,
                        sy1 + 9,
                        text=label[0],
                        fill="#495057",
                        font=PROMO_LABEL_FONT,
                        tags="promo_overlay",
                    )

    def _update_status_text(self, board: chess.Board | None = None) -> None:
        if self.controller.is_terminal():
            res = self.controller.result()
            winner_str = (
                "White wins"
                if res.winner == chess.WHITE
                else ("Black wins" if res.winner == chess.BLACK else "Draw")
            )
            self.status_var.set(f"Game Over: {res.result} — {winner_str} ({res.termination})")
            return

        if board is None:
            board = self.controller.board
        side = "White" if self.controller.turn == chess.WHITE else "Black"
        actor = self.player_names[self.controller.turn]
        check_suffix = " [CHECK!]" if board.is_check() else ""
        if self.ai_thinking:
            self.status_var.set(f"{side} ({actor}) is thinking...{check_suffix}")
        elif self.paused:
            self.status_var.set(f"Paused — {side} ({actor}) to move{check_suffix}")
        else:
            self.status_var.set(f"{side} ({actor}) to move{check_suffix}")

    # ============================================================
    # Human Interaction & Promotion
    # ============================================================

    def _get_promotion_targets(
        self,
    ) -> list[tuple[chess.Square, int, int, chess.PieceType, str]]:
        if self.pending_promotion is None:
            return []
        _, to_sq, _ = self.pending_promotion
        col, start_row = self.square_to_grid(to_sq)
        step = 1 if start_row < 4 else -1
        items: list[tuple[chess.Square, int, int, chess.PieceType, str]] = []
        for idx, (label, pt) in enumerate(PROMOTION_PIECES):
            row = start_row + idx * step
            if 0 <= row < 8:
                f_idx = 7 - col if self.flipped else col
                r_idx = row if self.flipped else 7 - row
                opt_sq = chess.square(f_idx, r_idx)
                items.append((opt_sq, col, row, pt, label))
        return items

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if self.pending_promotion is None:
            return
        sq = self.pixel_to_square(event.x, event.y)
        valid_sqs = {opt_sq for opt_sq, _, _, _, _ in self._get_promotion_targets()}
        new_hover = sq if sq in valid_sqs else None
        if new_hover != self.hover_promo_sq:
            self.hover_promo_sq = new_hover
            self._draw_board()

    def _on_canvas_click(self, event: tk.Event) -> None:
        square = self.pixel_to_square(event.x, event.y)
        if square is not None:
            self.select_square(square)
        elif self.pending_promotion is not None:
            self.pending_promotion = None
            self.hover_promo_sq = None
            self._refresh_ui()

    def select_square(
        self,
        square: chess.Square,
        *,
        promotion: chess.PieceType | None = None,
    ) -> bool:
        """Handle clicking a board square during a human turn. Return True if a move was made."""
        if (
            self.controller.is_terminal()
            or self.ai_thinking
            or (self.ai_worker_thread is not None and self.ai_worker_thread.is_alive())
        ):
            return False
        if self.current_agent() is not None:
            # Ignore human board clicks during AI's turn
            return False

        # 0. If Chess.com-style in-board promotion overlay is open, resolve or cancel it
        if self.pending_promotion is not None:
            _, _, candidate_moves = self.pending_promotion
            promo_map = {opt_sq: pt for opt_sq, _, _, pt, _ in self._get_promotion_targets()}
            self.pending_promotion = None
            self.hover_promo_sq = None
            if square in promo_map:
                chosen_pt = promo_map[square]
                matching = [m for m in candidate_moves if m.promotion == chosen_pt]
                chosen_move = matching[0] if matching else candidate_moves[0]
                self._apply_move(chosen_move, actor_name="Human", elapsed=0.0, stats=None)
                self._maybe_schedule_ai()
                return True
            self.selected_square = None
            self.legal_targets = {}
            self._refresh_ui()
            return False

        board = self.controller.board

        # 1. If square is a legal destination for the currently selected piece, make the move
        if self.selected_square is not None and square in self.legal_targets:
            candidate_moves = self.legal_targets[square]
            chosen_move = candidate_moves[0]
            if len(candidate_moves) > 1 or chosen_move.promotion is not None:
                if promotion is not None or self.promotion_chooser is not None:
                    promo_piece = (
                        promotion
                        if promotion is not None
                        else self.promotion_chooser(self.controller.turn)
                    )
                    matching = [m for m in candidate_moves if m.promotion == promo_piece]
                    chosen_move = matching[0] if matching else candidate_moves[0]
                else:
                    # Open Chess.com-style in-board promotion selector column
                    self.pending_promotion = (self.selected_square, square, candidate_moves)
                    self.hover_promo_sq = square
                    self._refresh_ui()
                    return False

            self._apply_move(chosen_move, actor_name="Human", elapsed=0.0, stats=None)
            self._maybe_schedule_ai()
            return True

        # 2. If clicking own piece, select it and compute legal target squares
        piece = board.piece_at(square)
        if piece is not None and piece.color == self.controller.turn:
            self.selected_square = square
            targets: dict[chess.Square, list[chess.Move]] = {}
            for move in self.controller.legal_moves:
                if move.from_square == square:
                    targets.setdefault(move.to_square, []).append(move)
            self.legal_targets = targets
            self._refresh_ui()
            return False

        # 3. Otherwise clear selection
        self.selected_square = None
        self.legal_targets = {}
        self._refresh_ui()
        return False

    def _apply_move(
        self,
        move: chess.Move,
        *,
        actor_name: str,
        elapsed: float,
        stats: dict[str, int] | None,
    ) -> None:
        board_before = self.controller.board
        san = board_before.san(move)
        ply = len(board_before.move_stack) + 1
        move_no = (ply + 1) // 2
        prefix = f"{move_no}." if board_before.turn == chess.WHITE else f"{move_no}..."

        self.controller.push(move)
        self.last_move = move
        self.selected_square = None
        self.legal_targets = {}
        self.pending_promotion = None
        self.hover_promo_sq = None

        if stats is not None:
            stat_str = format_search_stats(stats, elapsed)
            self.stats_var.set(f"{actor_name} ({san}): {stat_str}")
            log_entry = f"{prefix:<5} {san:<7} ({move.uci()}) [{stat_str}]"
        else:
            self.stats_var.set(f"{actor_name} played {san} ({move.uci()})")
            log_entry = f"{prefix:<5} {san:<7} ({move.uci()}) [Human]"

        self.move_log.append(log_entry)
        self.history_list.insert(tk.END, log_entry)
        self.history_list.yview_moveto(1.0)
        self._refresh_ui()

    # ============================================================
    # Asynchronous AI Execution (Non-blocking UI & Live AI vs AI)
    # ============================================================

    def _maybe_schedule_ai(self) -> None:
        if self.paused or self.controller.is_terminal():
            return
        if self.current_agent() is None:
            return
        if self.ai_thinking or (self.ai_worker_thread is not None and self.ai_worker_thread.is_alive()):
            if self.after_id is None:
                self.after_id = self.root.after(40, self._maybe_schedule_ai)
            return
        delay = self._parse_delay_ms() if self.last_move is not None else 20
        self.after_id = self.root.after(max(1, delay), self._start_ai_turn)

    def _start_ai_turn(self) -> None:
        self.after_id = None
        if self.controller.is_terminal():
            return
        if self.ai_thinking or (self.ai_worker_thread is not None and self.ai_worker_thread.is_alive()):
            return
        agent = self.current_agent()
        if agent is None:
            return

        self.search_token += 1
        token = self.search_token
        self.ai_thinking = True
        snapshot = self.controller.board
        self._update_status_text()

        def worker() -> None:
            start = perf_counter()
            try:
                move = agent.choose_move(snapshot)
                elapsed = perf_counter() - start
                raw_stats = getattr(agent, "last_search_stats", {})
                stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}
                self.ai_queue.put((token, move, elapsed, stats, None))
            except Exception as exc:
                elapsed = perf_counter() - start
                self.ai_queue.put((token, None, elapsed, {}, exc))

        self.ai_worker_thread = threading.Thread(target=worker, daemon=True)
        self.ai_worker_thread.start()

    def _poll_ai_queue(self) -> None:
        self.poll_after_id = None
        self.drain_ai_queue()
        try:
            if self.root.winfo_exists():
                self.poll_after_id = self.root.after(40, self._poll_ai_queue)
        except tk.TclError:
            return

    def drain_ai_queue(self) -> bool:
        """Process any completed AI search in the queue. Return True if a move was applied."""
        applied = False
        while True:
            try:
                token, move, elapsed, stats, error = self.ai_queue.get_nowait()
            except queue.Empty:
                break

            if token != self.search_token:
                # Discard stale result from a cancelled/reset search
                if self.ai_worker_thread is not None and not self.ai_worker_thread.is_alive():
                    self.ai_worker_thread = None
                    self.ai_thinking = False
                continue

            self.ai_thinking = False
            self.ai_worker_thread = None
            if error is not None or move is None:
                if isinstance(error, InterruptedError):
                    continue
                self.paused = True
                self.pause_btn_text.set("Resume")
                self._report_error("AI Error", str(error or "AI returned no move"))
                continue

            actor_name = self.player_names[self.controller.turn]
            try:
                self._apply_move(move, actor_name=actor_name, elapsed=elapsed, stats=stats)
                applied = True
            except Exception as exc:
                self.paused = True
                self.pause_btn_text.set("Resume")
                self._report_error("Illegal AI Move", str(exc))
                continue

            self._maybe_schedule_ai()
        return applied

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self._cancel_pending_ai()
        if self.poll_after_id is not None:
            try:
                self.root.after_cancel(self.poll_after_id)
            except tk.TclError:
                pass
            self.poll_after_id = None
        if self.owns_root:
            try:
                self.root.destroy()
            except tk.TclError:
                pass


def launch_gui(**kwargs) -> None:
    app = ChessGUI(**kwargs)
    app.run()


if __name__ == "__main__":
    launch_gui()
