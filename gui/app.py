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
from algorithms.evaluation import MATE_SCORE, evaluate
from algorithms.mcts import MCTS
from chess_system.ai_interface import ChessAI
from chess_system.game_controller import GameController

UNICODE_PIECES: dict[tuple[chess.Color, chess.PieceType], str] = {
    (chess.WHITE, chess.KING): "♔",
    (chess.WHITE, chess.QUEEN): "♕",
    (chess.WHITE, chess.ROOK): "♖",
    (chess.WHITE, chess.BISHOP): "♗",
    (chess.WHITE, chess.KNIGHT): "♘",
    (chess.WHITE, chess.PAWN): "♙",
    (chess.BLACK, chess.KING): "♚",
    (chess.BLACK, chess.QUEEN): "♛",
    (chess.BLACK, chess.ROOK): "♜",
    (chess.BLACK, chess.BISHOP): "♝",
    (chess.BLACK, chess.KNIGHT): "♞",
    (chess.BLACK, chess.PAWN): "♟",
}

PROMOTION_PIECES: tuple[tuple[str, chess.PieceType], ...] = (
    ("Queen (♛)", chess.QUEEN),
    ("Rook (♜)", chess.ROOK),
    ("Bishop (♝)", chess.BISHOP),
    ("Knight (♞)", chess.KNIGHT),
)

PIECE_FONT: tuple[str, int] = ("Segoe UI Symbol", 32)
COORD_FONT: tuple[str, int, str] = ("Segoe UI", 9, "bold")
EVAL_FONT: tuple[str, int, str] = ("Segoe UI", 8, "bold")


def compute_eval_bar_state(
    board: chess.Board,
    *,
    claim_draw: bool = True,
) -> tuple[float, str, int]:
    """Return (white_ratio in [0.0, 1.0], display_label, raw_centipawn_score)."""
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is not None:
        if outcome.winner == chess.WHITE:
            return 1.0, "1-0", MATE_SCORE
        if outcome.winner == chess.BLACK:
            return 0.0, "0-1", -MATE_SCORE
        return 0.5, "0.0", 0

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
    rollout_max_plies: int = 24,
    seed: int | None = None,
    claim_draw: bool = True,
) -> ChessAI | None:
    """Create a ChessAI instance from user/CLI parameters, or None for Human."""
    normalized = kind.strip().lower().replace("-", "_")
    if normalized == "human":
        return None
    if normalized in ("alpha_beta", "alphabeta"):
        parsed_depth = parse_optional_int(depth, "Alpha-Beta depth")
        if parsed_depth is None:
            raise ValueError("Alpha-Beta depth is required")
        return AlphaBetaAI(depth=parsed_depth)
    if normalized == "mcts":
        parsed_iter = parse_optional_int(iterations, "MCTS simulations")
        parsed_time = parse_optional_float(time_limit, "MCTS time budget")
        if parsed_iter is None and parsed_time is None:
            raise ValueError("MCTS requires simulations and/or time budget")
        return MCTS(
            iterations=parsed_iter,
            time_limit=parsed_time,
            rollout_max_plies=rollout_max_plies,
            seed=seed,
            claim_draw=claim_draw,
        )
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

    SQ_SIZE = 64
    MARGIN = 24
    BOARD_PX = SQ_SIZE * 8 + MARGIN * 2
    EVAL_WIDTH = 38

    LIGHT_SQ = "#eeeed2"
    DARK_SQ = "#769656"
    SELECTED_SQ = "#f6f669"
    LAST_MOVE_LIGHT = "#f5f682"
    LAST_MOVE_DARK = "#b9ca43"
    CHECK_SQ = "#e65c5c"
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
        move_delay_ms: int = 300,
        show_eval_bar: bool = True,
        auto_start: bool = True,
        promotion_chooser: Callable[[chess.Color], chess.PieceType] | None = None,
        error_handler: Callable[[str, str], None] | None = None,
    ):
        self.owns_root = root is None
        self.root = root if root is not None else tk.Tk()
        self.root.title("Chess Engine - Alpha-Beta & MCTS")
        self.root.configure(bg="#2b2b2b")

        self.promotion_chooser = promotion_chooser or self._prompt_promotion_dialog
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
        self.last_move: chess.Move | None = None
        self.flipped: bool = False
        self.paused: bool = False
        self.ai_thinking: bool = False
        self.search_token: int = 0
        self.after_id: str | None = None
        self.poll_after_id: str | None = None
        self.ai_queue: queue.Queue[tuple[int, chess.Move | None, float, dict, Exception | None]] = (
            queue.Queue()
        )
        self.move_log: list[str] = []

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
        self.delay_var = tk.StringVar(value=str(max(0, int(move_delay_ms))))
        self.fen_var = tk.StringVar(value=fen)
        self.show_eval_var = tk.BooleanVar(value=show_eval_bar)

        self.status_var = tk.StringVar(value="Ready")
        self.eval_text_var = tk.StringVar(value="Eval: 0.0")
        self.stats_var = tk.StringVar(value="Stats: -")
        self.pause_btn_text = tk.StringVar(value="Pause")

        self._build_ui()
        self.toggle_eval_bar(show_eval_bar)
        self.new_game(auto_start=auto_start)
        self.poll_after_id = self.root.after(25, self._poll_ai_queue)

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Left Area: Eval Bar (Col 0) + 8x8 Board Canvas (Col 1)
        self.board_container = ttk.Frame(main_frame)
        self.board_container.grid(row=0, column=0, padx=(0, 12), sticky="n")

        self.eval_canvas = tk.Canvas(
            self.board_container,
            width=self.EVAL_WIDTH,
            height=self.BOARD_PX,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.eval_canvas.grid(row=0, column=0, padx=(0, 6))

        self.board_canvas = tk.Canvas(
            self.board_container,
            width=self.BOARD_PX,
            height=self.BOARD_PX,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.board_canvas.grid(row=0, column=1)
        self.board_canvas.bind("<Button-1>", self._on_canvas_click)

        # Right Area: Controls, Config, Status, Move Log
        ctrl = ttk.Frame(main_frame)
        ctrl.grid(row=0, column=1, sticky="nsew")

        # 1. Quick Mode Presets
        preset_box = ttk.LabelFrame(ctrl, text="Quick Mode Presets", padding=6)
        preset_box.pack(fill="x", pady=(0, 6))
        ttk.Button(
            preset_box,
            text="Human vs AI",
            command=lambda: self.apply_preset("human", "alpha_beta"),
        ).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(
            preset_box,
            text="AI vs Human",
            command=lambda: self.apply_preset("mcts", "human"),
        ).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(
            preset_box,
            text="Live AI vs AI",
            command=lambda: self.apply_preset("alpha_beta", "mcts"),
        ).grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        ttk.Button(
            preset_box,
            text="Human vs Human",
            command=lambda: self.apply_preset("human", "human"),
        ).grid(row=0, column=3, padx=2, pady=2, sticky="ew")
        for col in range(4):
            preset_box.columnconfigure(col, weight=1)

        # 2. Player & AI Configuration
        cfg_box = ttk.LabelFrame(ctrl, text="Players & Search Budget", padding=6)
        cfg_box.pack(fill="x", pady=(0, 6))
        headers = ("Side", "Agent", "AB Depth", "MCTS Sims", "Time (s)")
        for col, text in enumerate(headers):
            ttk.Label(cfg_box, text=text, font=("Segoe UI", 9, "bold")).grid(
                row=0, column=col, padx=3, pady=2
            )

        self._build_side_row(
            cfg_box,
            row=1,
            label="White (♔)",
            kind_var=self.white_kind_var,
            depth_var=self.white_depth_var,
            iter_var=self.white_iter_var,
            time_var=self.white_time_var,
        )
        self._build_side_row(
            cfg_box,
            row=2,
            label="Black (♚)",
            kind_var=self.black_kind_var,
            depth_var=self.black_depth_var,
            iter_var=self.black_iter_var,
            time_var=self.black_time_var,
        )

        # FEN & Delay Row
        fen_frame = ttk.Frame(cfg_box)
        fen_frame.grid(row=3, column=0, columnspan=5, sticky="ew", pady=(6, 0))
        ttk.Label(fen_frame, text="FEN:").pack(side="left")
        ttk.Entry(fen_frame, textvariable=self.fen_var, width=32).pack(
            side="left", padx=4, fill="x", expand=True
        )
        ttk.Button(
            fen_frame,
            text="Start FEN",
            width=9,
            command=lambda: self.fen_var.set(chess.STARTING_FEN),
        ).pack(side="left")

        # 3. Live Match Controls & View Options
        action_box = ttk.LabelFrame(ctrl, text="Match & View Controls", padding=6)
        action_box.pack(fill="x", pady=(0, 6))

        btn_row = ttk.Frame(action_box)
        btn_row.pack(fill="x", pady=2)
        ttk.Button(btn_row, text="New Game", command=self.new_game).pack(
            side="left", padx=2, expand=True, fill="x"
        )
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

        opt_row = ttk.Frame(action_box)
        opt_row.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(
            opt_row,
            text="Show Eval Bar (Left)",
            variable=self.show_eval_var,
            command=lambda: self.toggle_eval_bar(self.show_eval_var.get()),
        ).pack(side="left", padx=(2, 12))

        ttk.Label(opt_row, text="AI Move Delay (ms):").pack(side="left")
        ttk.Spinbox(
            opt_row,
            from_=0,
            to=3000,
            increment=100,
            textvariable=self.delay_var,
            width=7,
        ).pack(side="left", padx=4)

        # 4. Status & Search Metrics
        status_box = ttk.LabelFrame(ctrl, text="Game Status & AI Telemetry", padding=6)
        status_box.pack(fill="x", pady=(0, 6))
        ttk.Label(
            status_box,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
            wraplength=360,
        ).pack(anchor="w")
        ttk.Label(
            status_box,
            textvariable=self.eval_text_var,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            status_box,
            textvariable=self.stats_var,
            font=("Consolas", 9),
            wraplength=360,
        ).pack(anchor="w", pady=(2, 0))

        # 5. Live Move History
        hist_box = ttk.LabelFrame(ctrl, text="Move History", padding=6)
        hist_box.pack(fill="both", expand=True)
        self.history_list = tk.Listbox(
            hist_box, font=("Consolas", 9), height=12, activestyle="none"
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
        row: int,
        label: str,
        kind_var: tk.StringVar,
        depth_var: tk.StringVar,
        iter_var: tk.StringVar,
        time_var: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=3, pady=2, sticky="w")
        combo = ttk.Combobox(
            parent,
            textvariable=kind_var,
            values=("human", "alpha_beta", "mcts"),
            state="readonly",
            width=11,
        )
        combo.grid(row=row, column=1, padx=3, pady=2)
        ttk.Spinbox(parent, from_=1, to=8, textvariable=depth_var, width=6).grid(
            row=row, column=2, padx=3, pady=2
        )
        ttk.Spinbox(
            parent, from_=1, to=20000, increment=50, textvariable=iter_var, width=8
        ).grid(row=row, column=3, padx=3, pady=2)
        ttk.Entry(parent, textvariable=time_var, width=7).grid(
            row=row, column=4, padx=3, pady=2
        )

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

    def apply_preset(self, white_kind: str, black_kind: str, *, auto_start: bool = True) -> None:
        self.white_kind_var.set(white_kind)
        self.black_kind_var.set(black_kind)
        self.new_game(auto_start=auto_start)

    def _parse_delay_ms(self) -> int:
        try:
            return max(0, int(float(self.delay_var.get().strip() or "0")))
        except ValueError:
            return 500

    def _cancel_pending_ai(self) -> None:
        self.search_token += 1
        self.ai_thinking = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

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
                claim_draw=new_controller.claim_draw,
            )
            black_agent = build_agent(
                self.black_kind_var.get(),
                depth=self.black_depth_var.get(),
                iterations=self.black_iter_var.get(),
                time_limit=self.black_time_var.get(),
                claim_draw=new_controller.claim_draw,
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
            chess.WHITE: self._format_agent_label(self.white_kind_var.get()),
            chess.BLACK: self._format_agent_label(self.black_kind_var.get()),
        }
        self.selected_square = None
        self.legal_targets = {}
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
    def _format_agent_label(kind: str) -> str:
        norm = kind.strip().lower().replace("-", "_")
        if norm == "human":
            return "Human"
        if norm in ("alpha_beta", "alphabeta"):
            return "Alpha-Beta"
        if norm == "mcts":
            return "MCTS"
        return kind

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
        if self.controller.is_terminal() or self.current_agent() is None or self.ai_thinking:
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
        white_ratio, label, cp = compute_eval_bar_state(
            board, claim_draw=self.controller.claim_draw
        )
        side_adv = (
            "Equal"
            if cp == 0
            else ("White advantage" if cp > 0 else "Black advantage")
        )
        self.eval_text_var.set(f"Eval: {label} ({side_adv})")

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

            # Piece glyph
            piece = board.piece_at(square)
            if piece is not None:
                glyph = UNICODE_PIECES[(piece.color, piece.piece_type)]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if piece.color == chess.WHITE:
                    # Single subtle drop shadow for white piece contrast on light squares
                    self.board_canvas.create_text(
                        cx + 1,
                        cy + 1,
                        text=glyph,
                        fill="#333333",
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
                    self.board_canvas.create_text(
                        cx,
                        cy,
                        text=glyph,
                        fill="#1a1a1a",
                        font=PIECE_FONT,
                        tags="piece",
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
                fill="#cccccc",
                font=COORD_FONT,
            )
            self.board_canvas.create_text(
                self.MARGIN // 2,
                ry,
                text=rank_char,
                fill="#cccccc",
                font=COORD_FONT,
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

    def _on_canvas_click(self, event: tk.Event) -> None:
        square = self.pixel_to_square(event.x, event.y)
        if square is not None:
            self.select_square(square)

    def select_square(
        self,
        square: chess.Square,
        *,
        promotion: chess.PieceType | None = None,
    ) -> bool:
        """Handle clicking a board square during a human turn. Return True if a move was made."""
        if self.controller.is_terminal() or self.ai_thinking:
            return False
        if self.current_agent() is not None:
            # Ignore human board clicks during AI's turn
            return False

        board = self.controller.board

        # 1. If square is a legal destination for the currently selected piece, make the move
        if self.selected_square is not None and square in self.legal_targets:
            candidate_moves = self.legal_targets[square]
            chosen_move = candidate_moves[0]
            if len(candidate_moves) > 1 or chosen_move.promotion is not None:
                promo_piece = (
                    promotion
                    if promotion is not None
                    else self.promotion_chooser(self.controller.turn)
                )
                matching = [m for m in candidate_moves if m.promotion == promo_piece]
                chosen_move = matching[0] if matching else candidate_moves[0]

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

    def _prompt_promotion_dialog(self, color: chess.Color) -> chess.PieceType:
        dialog = tk.Toplevel(self.root)
        dialog.title("Promote Pawn")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        choice = tk.IntVar(value=chess.QUEEN)
        ttk.Label(
            dialog,
            text="Choose promotion piece:",
            font=("Segoe UI", 10, "bold"),
            padding=8,
        ).pack()

        btn_frame = ttk.Frame(dialog, padding=8)
        btn_frame.pack()
        for label, piece_type in PROMOTION_PIECES:
            glyph = UNICODE_PIECES[(color, piece_type)]
            ttk.Button(
                btn_frame,
                text=f"{glyph} {label}",
                width=16,
                command=lambda pt=piece_type: (choice.set(pt), dialog.destroy()),
            ).pack(pady=2)

        dialog.wait_window()
        return int(choice.get())

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
        if self.paused or self.controller.is_terminal() or self.ai_thinking:
            return
        if self.current_agent() is None:
            return
        delay = self._parse_delay_ms() if self.last_move is not None else 20
        self.after_id = self.root.after(max(1, delay), self._start_ai_turn)

    def _start_ai_turn(self) -> None:
        self.after_id = None
        if self.controller.is_terminal() or self.ai_thinking:
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

        threading.Thread(target=worker, daemon=True).start()

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
                continue

            self.ai_thinking = False
            if error is not None or move is None:
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
