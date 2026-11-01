"""Entrypoint for the Chess Engine GUI, Live AI vs AI viewer, and CLI experiments."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import chess

from experiments.benchmark import run_match, save_match
from experiments.tournament import run_tournament, summarize
from gui.app import build_agent, launch_gui


def _add_gui_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--white",
        default="human",
        choices=("human", "alpha_beta", "alpha-beta", "mcts"),
        help="White player type (default: human)",
    )
    parser.add_argument(
        "--black",
        default="alpha_beta",
        choices=("human", "alpha_beta", "alpha-beta", "mcts"),
        help="Black player type (default: alpha_beta)",
    )
    parser.add_argument("--white-depth", type=int, default=3, help="White Alpha-Beta depth")
    parser.add_argument("--black-depth", type=int, default=3, help="Black Alpha-Beta depth")
    parser.add_argument(
        "--white-iterations", type=int, default=100, help="White MCTS simulations"
    )
    parser.add_argument(
        "--black-iterations", type=int, default=100, help="Black MCTS simulations"
    )
    parser.add_argument(
        "--white-time", type=float, default=None, help="White MCTS time budget (seconds)"
    )
    parser.add_argument(
        "--black-time", type=float, default=None, help="Black MCTS time budget (seconds)"
    )
    parser.add_argument(
        "--delay", type=int, default=300, help="Live AI vs AI move delay in ms (default: 300)"
    )
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="Starting FEN position")
    parser.add_argument(
        "--no-eval-bar",
        action="store_true",
        help="Hide the left-side evaluation bar initially",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chess Engine: 8x8 Tkinter GUI (Human vs AI, Live AI vs AI) & Benchmark CLI"
    )
    _add_gui_args(parser)

    subparsers = parser.add_subparsers(dest="command")

    gui_parser = subparsers.add_parser("gui", help="Launch the 8x8 Chess GUI (default)")
    _add_gui_args(gui_parser)

    match_parser = subparsers.add_parser(
        "match", help="Run a headless AI vs AI match and print/save telemetry"
    )
    match_parser.add_argument(
        "--white",
        default="alpha_beta",
        choices=("alpha_beta", "alpha-beta", "mcts"),
    )
    match_parser.add_argument(
        "--black",
        default="mcts",
        choices=("alpha_beta", "alpha-beta", "mcts"),
    )
    match_parser.add_argument("--white-depth", type=int, default=3)
    match_parser.add_argument("--black-depth", type=int, default=3)
    match_parser.add_argument("--white-iterations", type=int, default=400)
    match_parser.add_argument("--black-iterations", type=int, default=400)
    match_parser.add_argument("--white-time", type=float, default=None)
    match_parser.add_argument("--black-time", type=float, default=None)
    match_parser.add_argument("--seed", type=int, default=42)
    match_parser.add_argument("--fen", default=chess.STARTING_FEN)
    match_parser.add_argument("--max-plies", type=int, default=200)
    match_parser.add_argument("--output-dir", default=None)
    match_parser.add_argument("--match-id", default="match_001")

    tourney_parser = subparsers.add_parser(
        "tournament",
        help="Run a color-swapped tournament between Alpha-Beta and MCTS for report data",
    )
    tourney_parser.add_argument("--depth", type=int, default=3, help="Alpha-Beta depth")
    tourney_parser.add_argument(
        "--iterations", type=int, default=400, help="MCTS simulations"
    )
    tourney_parser.add_argument(
        "--time-limit", type=float, default=None, help="MCTS time budget (seconds)"
    )
    tourney_parser.add_argument("--seed", type=int, default=42, help="MCTS random seed")
    tourney_parser.add_argument("--rounds", type=int, default=1, help="Paired rounds")
    tourney_parser.add_argument("--max-plies", type=int, default=200, help="Max plies/game")
    tourney_parser.add_argument(
        "--eval-mode", type=str, default="hybrid",
        choices=("ab_classic", "mcts_pe", "hybrid"),
        help="MCTS evaluation mode (default: hybrid)"
    )
    tourney_parser.add_argument(
        "--output-dir", default=None, help="New directory to save PGN/CSV/JSON/standings.csv"
    )

    return parser


def _run_cli_match(args: argparse.Namespace) -> int:
    white_ai = build_agent(
        args.white,
        depth=args.white_depth,
        iterations=args.white_iterations,
        time_limit=args.white_time,
        seed=args.seed,
    )
    black_ai = build_agent(
        args.black,
        depth=args.black_depth,
        iterations=args.black_iterations,
        time_limit=args.black_time,
        seed=args.seed,
    )
    assert white_ai is not None and black_ai is not None
    white_name = f"White({args.white})"
    black_name = f"Black({args.black})"
    result = run_match(
        white_ai,
        black_ai,
        white_name=white_name,
        black_name=black_name,
        fen=args.fen,
        max_plies=args.max_plies,
        config={
            "white": {
                "kind": args.white,
                "depth": args.white_depth,
                "iterations": args.white_iterations,
                "time_limit": args.white_time,
            },
            "black": {
                "kind": args.black,
                "depth": args.black_depth,
                "iterations": args.black_iterations,
                "time_limit": args.black_time,
            },
        },
    )
    if args.output_dir:
        save_match(result, args.output_dir, args.match_id)
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
    return 0


def _run_cli_tournament(args: argparse.Namespace) -> int:
    ab_cfg = {"depth": args.depth}
    mcts_cfg = {
        "iterations": args.iterations,
        "time_limit": args.time_limit,
        "seed": args.seed,
        "eval_mode": args.eval_mode,
    }
    matches = run_tournament(
        factory_a=lambda: build_agent("alpha_beta", depth=args.depth),  # type: ignore[return-value]
        factory_b=lambda: build_agent(  # type: ignore[return-value]
            "mcts",
            iterations=args.iterations,
            time_limit=args.time_limit,
            seed=args.seed,
            eval_mode=args.eval_mode,
        ),
        name_a="Alpha-Beta",
        name_b="MCTS",
        rounds=args.rounds,
        max_plies=args.max_plies,
        config={"Alpha-Beta": ab_cfg, "MCTS": mcts_cfg},
        output_dir=args.output_dir,
    )
    standings = summarize(matches)
    print(json.dumps(standings, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "match":
        return _run_cli_match(args)
    if args.command == "tournament":
        return _run_cli_tournament(args)

    launch_gui(
        fen=args.fen,
        white_kind=args.white.replace("-", "_"),
        black_kind=args.black.replace("-", "_"),
        white_depth=args.white_depth,
        black_depth=args.black_depth,
        white_iterations=args.white_iterations,
        black_iterations=args.black_iterations,
        white_time_limit=args.white_time,
        black_time_limit=args.black_time,
        move_delay_ms=args.delay,
        show_eval_bar=not args.no_eval_bar,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
