"""Reproducible MCTS parameter sweep; run with python -m experiments.mcts_tuning."""

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import chess
from algorithms.mcts import MCTS
from experiments.tournament import run_tournament, summarize


def run_sweep(
    output_dir: str | Path,
    *,
    constants=(0.5, math.sqrt(2), 2.0),
    budgets=(32, 128),
    seeds=(7, 19, 42),
    rollout_max_plies=24,
    matches=False,
    max_plies=40,
):
    """Measure mate-in-one success and latency; optionally play paired self-play.

    Opponent in self-play is MCTS with sqrt(2), the same iteration/rollout budget
    and seed. This is parameter tuning, not an Alpha-Beta comparison.
    """
    constants, budgets, seeds = tuple(constants), tuple(budgets), tuple(seeds)
    if not constants or not budgets or not seeds:
        raise ValueError("Nonempty constants, budgets and seeds are required")
    if type(max_plies) is not int or max_plies < 1:
        raise ValueError("max_plies must be a positive integer")
    for constant in constants:
        for budget in budgets:
            for seed in seeds:
                MCTS(budget, exploration_constant=constant,
                     rollout_max_plies=rollout_max_plies, seed=seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    white_mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    positions = {"mate_white": white_mate, "mate_black": white_mate.mirror(),
                 "starting": chess.Board()}
    metadata = dict(
        python=platform.python_version(), platform=platform.platform(),
        chess=chess.__version__, constants=constants, budgets=budgets, seeds=seeds,
        rollout_max_plies=rollout_max_plies, claim_draw=True,
        matches=matches, max_plies=max_plies,
        positions={name: board.fen() for name, board in positions.items()},
        reference="MCTS c=sqrt(2), matching iteration budget; not Alpha-Beta",
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    rows = []
    with (output / "searches.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "constant", "budget", "seed", "position", "move", "seconds",
            "iterations", "rollouts", "mate_in_one",
        ])
        writer.writeheader()
        for ci, constant in enumerate(constants):
            for budget in budgets:
                for seed in seeds:
                    for name, board in positions.items():
                        ai = MCTS(budget, exploration_constant=constant,
                                   rollout_max_plies=rollout_max_plies, seed=seed)
                        start = perf_counter()
                        move = ai.choose_move(board)
                        seconds = perf_counter() - start
                        after = board.copy()
                        after.push(move)
                        row = dict(constant=constant, budget=budget, seed=seed,
                                   position=name, move=move.uci(), seconds=seconds,
                                   **ai.last_search_stats,
                                   mate_in_one=int(after.is_checkmate()) if name != "starting" else "")
                        rows.append(row)
                        writer.writerow(row)
                    if matches:
                        candidate = dict(iterations=budget, exploration_constant=constant,
                                         rollout_max_plies=rollout_max_plies, seed=seed)
                        reference = dict(candidate, exploration_constant=math.sqrt(2))
                        games = run_tournament(
                            lambda: MCTS(**candidate), lambda: MCTS(**reference),
                            name_a="candidate", name_b="reference", max_plies=max_plies,
                            config={"candidate": candidate, "reference": reference},
                            output_dir=output / f"c_{ci}_budget_{budget}_seed_{seed}",
                        )
                        print(json.dumps({"constant": constant, "budget": budget,
                                          "seed": seed, "standings": summarize(games)}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--constants", nargs="+", type=float, default=[0.5, math.sqrt(2), 2.0])
    parser.add_argument("--budgets", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 42])
    parser.add_argument("--rollout-plies", type=int, default=24)
    parser.add_argument("--matches", action="store_true", help="Also run color-swapped self-play")
    parser.add_argument("--max-plies", type=int, default=40)
    args = parser.parse_args()
    rows = run_sweep(args.output, constants=args.constants, budgets=args.budgets,
                     seeds=args.seeds, rollout_max_plies=args.rollout_plies,
                     matches=args.matches, max_plies=args.max_plies)
    print(f"Saved {len(rows)} searches to {args.output}")


if __name__ == "__main__":
    main()
