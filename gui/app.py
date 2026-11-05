"""Tkinter 8x8 Chess GUI with Chess.com-style Eval Bar, Human vs AI, and Live AI vs AI."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from time import perf_counter
from tkinter import messagebox, ttk
from typing import Callable

import chess

from chess_system.ai_interface import ChessAI
from chess_system.game_controller import GameController
from gui.agent_factory import (
    build_agent,
    format_search_stats,
    parse_optional_float,
    parse_optional_int,
)
from gui.board_view import (
    draw_board,
    get_promotion_targets,
    pixel_to_square,
    square_to_grid,
)
from gui.eval_bar import compute_eval_bar_state, render_eval_bar
from gui.theme import (
    BOARD_PX,
    COORD_FONT,
    EVAL_FONT,
    EVAL_WIDTH,
    PIECE_FONT,
    PROMO_LABEL_FONT,
    PROMOTION_PIECES,
    SQ_SIZE,
    UNICODE_PIECES,
    setup_theme,
)

__all__ = [
    "ChessGUI",
    "UNICODE_PIECES",
    "PROMOTION_PIECES",
    "PIECE_FONT",
    "COORD_FONT",
    "EVAL_FONT",
    "PROMO_LABEL_FONT",
    "build_agent",
    "compute_eval_bar_state",
    "format_search_stats",
    "parse_optional_int",
    "parse_optional_float",
    "launch_gui",
]


class ChessGUI:
    """Interactive 8x8 Chess GUI supporting Human vs AI and Live AI vs AI."""

    SQ_SIZE = SQ_SIZE
    MARGIN = 26
    BOARD_PX = BOARD_PX
    EVAL_WIDTH = EVAL_WIDTH

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
    ) -> None:
        self.owns_root = root is None
        self.root = tk.Tk() if self.owns_root else root
        if self.owns_root:
            self.root.title("Chess Engine — 8x8 GUI (Human vs AI & Live AI vs AI)")
            self.root.geometry("1060x680")
            self.root.minsize(980, 640)

        self.controller = GameController(fen, claim_draw=True)
        self.promotion_chooser = promotion_chooser
        self.error_handler = error_handler or (lambda title, msg: messagebox.showerror(title, msg))
        self.last_error: str | None = None

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
        setup_theme(self.root)

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
    ) -> bool:
        self.white_kind_var.set(white_kind)
        self.black_kind_var.set(black_kind)
        if white_eval:
            self.white_eval_mode_var.set(white_eval)
        if black_eval:
            self.black_eval_mode_var.set(black_eval)
        return self.new_game(auto_start=auto_start)

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
        if current_is_human and other_is_ai and len(board_stack) >= 2:
            self.controller.pop()
            self.controller.pop()
            if len(self.move_log) >= 2:
                self.move_log.pop()
                self.move_log.pop()
        else:
            self.controller.pop()
            if self.move_log:
                self.move_log.pop()

        self.selected_square = None
        self.legal_targets = {}
        self.pending_promotion = None
        self.hover_promo_sq = None
        self.last_move = self.controller.board.peek() if self.controller.board.move_stack else None

        # Rebuild history listbox
        self.history_list.delete(0, tk.END)
        for line in self.move_log:
            self.history_list.insert(tk.END, line)
        if self.move_log:
            self.history_list.see(tk.END)

        self._refresh_ui()
        self._maybe_schedule_ai()
        return True

    # ============================================================
    # Board Coordinate Helpers & Rendering
    # ============================================================

    def square_to_grid(self, square: chess.Square) -> tuple[int, int]:
        return square_to_grid(square, self.flipped)

    def pixel_to_square(self, x: int, y: int) -> chess.Square | None:
        return pixel_to_square(x, y, self.flipped)

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
        render_eval_bar(
            self.eval_canvas,
            board,
            flipped=self.flipped,
            show_eval=self.show_eval_var.get(),
            eval_mode=mode_val,
            claim_draw=self.controller.claim_draw,
            eval_text_var=self.eval_text_var,
        )

    def _draw_board(self, board: chess.Board | None = None) -> None:
        if board is None:
            board = self.controller.board
        draw_board(
            self.board_canvas,
            board,
            flipped=self.flipped,
            selected_square=self.selected_square,
            legal_targets=self.legal_targets,
            last_move=self.last_move,
            pending_promotion=self.pending_promotion,
            hover_promo_sq=self.hover_promo_sq,
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
        return get_promotion_targets(self.pending_promotion, self.flipped)

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

        # In-board promotion piece choice click
        if self.pending_promotion is not None:
            from_sq, to_sq, promo_moves = self.pending_promotion
            promo_targets = self._get_promotion_targets()
            matched_pt: chess.PieceType | None = None
            for opt_sq, _, _, pt, _ in promo_targets:
                if square == opt_sq:
                    matched_pt = pt
                    break

            if matched_pt is not None:
                chosen_move = next(
                    (m for m in promo_moves if m.promotion == matched_pt),
                    None,
                )
                self.pending_promotion = None
                self.hover_promo_sq = None
                if chosen_move is not None:
                    self._apply_move(chosen_move, actor_name="Human")
                    self._maybe_schedule_ai()
                    return True
            else:
                # Clicked outside promo column -> dismiss popup
                self.pending_promotion = None
                self.hover_promo_sq = None
                self._refresh_ui()
                return False

        board = self.controller.board
        clicked_piece = board.piece_at(square)

        # 1. Clicked an existing selection -> deselect
        if self.selected_square == square:
            self.selected_square = None
            self.legal_targets = {}
            self._refresh_ui()
            return False

        # 2. Clicked a valid destination square
        if square in self.legal_targets:
            candidate_moves = self.legal_targets[square]
            self.selected_square = None
            self.legal_targets = {}

            # Promotion move handling
            if any(m.promotion is not None for m in candidate_moves):
                if promotion is not None:
                    chosen = next((m for m in candidate_moves if m.promotion == promotion), candidate_moves[0])
                    self._apply_move(chosen, actor_name="Human")
                    self._maybe_schedule_ai()
                    return True

                if self.promotion_chooser is not None:
                    chosen_pt = self.promotion_chooser(board.turn)
                    chosen = next((m for m in candidate_moves if m.promotion == chosen_pt), candidate_moves[0])
                    self._apply_move(chosen, actor_name="Human")
                    self._maybe_schedule_ai()
                    return True

                # Open modern Chess.com-style vertical column dialog directly inside canvas
                from_sq = candidate_moves[0].from_square
                self.pending_promotion = (from_sq, square, candidate_moves)
                self.hover_promo_sq = square
                self._refresh_ui()
                return False

            self._apply_move(candidate_moves[0], actor_name="Human")
            self._maybe_schedule_ai()
            return True

        # 3. Clicked a piece of the active player's color -> select it
        if clicked_piece is not None and clicked_piece.color == self.controller.turn:
            self.selected_square = square
            self.legal_targets = {}
            for move in self.controller.legal_moves:
                if move.from_square == square:
                    self.legal_targets.setdefault(move.to_square, []).append(move)
            self._refresh_ui()
            return False

        # 4. Clicked an empty/invalid square -> clear selection
        self.selected_square = None
        self.legal_targets = {}
        self._refresh_ui()
        return False

    def _apply_move(
        self,
        move: chess.Move,
        *,
        actor_name: str,
        elapsed: float | None = None,
        stats: dict[str, int] | None = None,
    ) -> None:
        turn_num = (len(self.controller.board.move_stack) // 2) + 1
        is_white = self.controller.turn == chess.WHITE
        prefix = f"{turn_num}. " if is_white else f"{turn_num}... "

        san = self.controller.board.san(move)
        stats_part = f" ({format_search_stats(stats, elapsed)})" if elapsed is not None else ""
        log_line = f"{prefix}{san:<7} [{actor_name}]{stats_part}"

        self.controller.push(move)
        self.last_move = move
        self.selected_square = None
        self.legal_targets = {}
        self.pending_promotion = None
        self.hover_promo_sq = None

        self.move_log.append(log_line)
        self.history_list.insert(tk.END, log_line)
        self.history_list.see(tk.END)

        if stats is not None and elapsed is not None:
            self.stats_var.set(f"Last Move ({actor_name}): {format_search_stats(stats, elapsed)}")

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
