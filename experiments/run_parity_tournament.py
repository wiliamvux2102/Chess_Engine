"""Run Parity Evaluation Batch Tournament.

Benchmarks Alpha-Beta and MCTS under identical evaluation heuristics:
- Suite 1: Both algorithms use ab_classic
- Suite 2: Both algorithms use mcts_pe
Parameters:
- Depths: 1, 2, 3
- Iterations: 50, 100, 200
- 5 rounds = 10 games per pair (color-swapped, unique seeds per round)
- max_plies: 500 (FIDE natural termination)
Total: 2 modes x 3 depths x 3 iters = 18 pairs (180 games).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _is_pair_complete_and_matching(
    output_dir: Path,
    *,
    eval_mode: str,
    depth: int,
    iterations: int,
    rounds: int,
    max_plies: int,
    seed: int,
) -> bool:
    """Verify that output_dir contains a complete tournament matching the exact parameters."""
    standings_path = output_dir / "standings.csv"
    if not standings_path.is_file():
        return False

    expected_games = rounds * 2
    try:
        with open(standings_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        if len(reader) != 2:
            return False
        for row in reader:
            if int(row.get("games", -1)) != expected_games:
                return False

        # Reject if extra rounds from a larger previous tournament exist
        if (output_dir / f"opening_1_round_{rounds + 1}_swap_0.json").exists():
            return False

        game_idx = 0
        for r in range(1, rounds + 1):
            for s in (0, 1):
                game_idx += 1
                match_json = output_dir / f"opening_1_round_{r}_swap_{s}.json"
                if not match_json.is_file():
                    return False
                data = json.loads(match_json.read_text(encoding="utf-8"))
                cfg = data.get("config", {})
                if cfg.get("max_plies") != max_plies:
                    return False
                ab_cfg = cfg.get("Alpha-Beta", {})
                if ab_cfg.get("depth") != depth or ab_cfg.get("eval_mode") != eval_mode:
                    return False
                mcts_cfg = cfg.get("MCTS", {})
                expected_seed = (seed + game_idx - 1) if seed is not None else None
                if (
                    mcts_cfg.get("iterations") != iterations
                    or mcts_cfg.get("eval_mode") != eval_mode
                    or mcts_cfg.get("seed") != expected_seed
                ):
                    return False
        return True
    except Exception:
        return False


def run_single_pair(
    eval_mode: str,
    depth: int,
    iterations: int,
    output_dir: Path,
    rounds: int = 5,
    max_plies: int = 500,
    seed: int = 42,
) -> tuple[str, int, int, float, bool, str]:
    """Run one tournament pair in a separate subprocess."""
    if _is_pair_complete_and_matching(
        output_dir,
        eval_mode=eval_mode,
        depth=depth,
        iterations=iterations,
        rounds=rounds,
        max_plies=max_plies,
        seed=seed,
    ):
        return eval_mode, depth, iterations, 0.0, True, "Already completed (config verified)"

    # Ensure output_dir is fresh when incomplete or config changed
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        "main.py",
        "tournament",
        "--depth", str(depth),
        "--iterations", str(iterations),
        "--ab-eval-mode", eval_mode,
        "--eval-mode", eval_mode,
        "--rounds", str(rounds),
        "--max-plies", str(max_plies),
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]

    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        return eval_mode, depth, iterations, elapsed, False, res.stderr.strip() or f"Exit code {res.returncode}"
    return eval_mode, depth, iterations, elapsed, True, "OK"


def aggregate_master_standings(base_dir: Path, pairs: list[tuple[str, int, int]]) -> Path:
    """Aggregate individual standings.csv files into master_standings.csv."""
    master_csv = base_dir / "master_standings.csv"
    fieldnames = [
        "eval_mode", "depth", "iterations", "agent",
        "games", "wins", "draws", "losses", "unfinished", "plies", "seconds"
    ]
    rows: list[dict[str, str | int | float]] = []

    for eval_mode, depth, iterations in pairs:
        pair_dir = base_dir / f"{eval_mode}_d{depth}_i{iterations}"
        standings_file = pair_dir / "standings.csv"
        if not standings_file.is_file():
            continue
        with open(standings_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "eval_mode": eval_mode,
                    "depth": depth,
                    "iterations": iterations,
                    "agent": row.get("agent", ""),
                    "games": int(row.get("games", 0)),
                    "wins": int(row.get("wins", 0)),
                    "draws": int(row.get("draws", 0)),
                    "losses": int(row.get("losses", 0)),
                    "unfinished": int(row.get("unfinished", 0)),
                    "plies": int(row.get("plies", 0)),
                    "seconds": float(row.get("seconds", 0.0)),
                })

    with open(master_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return master_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full evaluation-parity tournament batch.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent tournament workers (default: 4)")
    parser.add_argument("--rounds", type=int, default=5, help="Rounds per pair (default: 5 = 10 games)")
    parser.add_argument("--max-plies", type=int, default=500, help="Max half-moves per game (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/parity_eval_batch"),
        help="Target base directory",
    )
    args = parser.parse_args(argv)

    base_dir = args.output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    eval_modes = ["ab_classic", "mcts_pe"]
    depths = [1, 2, 3]
    iters = [50, 100, 200]

    pairs: list[tuple[str, int, int]] = [
        (mode, d, i)
        for mode in eval_modes
        for d in depths
        for i in iters
    ]

    total_pairs = len(pairs)
    print("=" * 60)
    print(f"=== PARITY TOURNAMENT BATCH ({total_pairs} pairs, {total_pairs * args.rounds * 2} games) ===")
    print(f"=== Workers: {args.workers} | Rounds: {args.rounds} ({args.rounds * 2} games/pair) ===")
    print(f"=== Output: {base_dir} ===")
    print("=" * 60)

    start_all = time.perf_counter()
    completed = 0
    failed_pairs: list[tuple[str, int, int, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for mode, d, i in pairs:
            pair_dir = base_dir / f"{mode}_d{d}_i{i}"
            fut = executor.submit(
                run_single_pair,
                mode, d, i,
                pair_dir,
                rounds=args.rounds,
                max_plies=args.max_plies,
                seed=args.seed,
            )
            futures[fut] = (mode, d, i)

        for fut in as_completed(futures):
            mode, d, i = futures[fut]
            completed += 1
            try:
                ret_mode, ret_d, ret_i, elapsed, ok, msg = fut.result()
                if ok:
                    mins = elapsed / 60.0
                    print(f"[{completed}/{total_pairs}] DONE {ret_mode} d={ret_d} i={ret_i} in {mins:.2f}m ({msg})")
                else:
                    failed_pairs.append((ret_mode, ret_d, ret_i, msg))
                    print(f"[{completed}/{total_pairs}] FAILED {ret_mode} d={ret_d} i={ret_i}: {msg}")
            except Exception as exc:
                failed_pairs.append((mode, d, i, str(exc)))
                print(f"[{completed}/{total_pairs}] ERROR {mode} d={d} i={i}: {exc}")

    total_mins = (time.perf_counter() - start_all) / 60.0
    print("=" * 60)
    master_csv = aggregate_master_standings(base_dir, pairs)

    if failed_pairs:
        print(f"=== BATCH INCOMPLETE: {len(failed_pairs)}/{total_pairs} PAIR(S) FAILED in {total_mins:.2f} minutes ===")
        for f_mode, f_d, f_i, f_msg in failed_pairs:
            print(f"  - {f_mode} d={f_d} i={f_i}: {f_msg}")
        print(f"=== Partial standings written to: {master_csv} ===")
        print("=" * 60)
        return 1

    print(f"=== ALL {total_pairs} PAIRS COMPLETED in {total_mins:.2f} minutes ===")
    print(f"=== Master standings written to: {master_csv} ===")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
