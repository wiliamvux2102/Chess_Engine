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
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


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
    standings_path = output_dir / "standings.csv"
    if standings_path.is_file():
        # Check if complete (header + 2 rows)
        try:
            with open(standings_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) >= 3:
                return eval_mode, depth, iterations, 0.0, True, "Already completed"
        except Exception:
            pass

    # Ensure output_dir is fresh
    if output_dir.exists():
        import shutil
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
        return eval_mode, depth, iterations, elapsed, False, res.stderr.strip()
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


def main() -> int:
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
    args = parser.parse_args()

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
    print(f"=== Workers: {args.workers} | Rounds: {args.rounds} (10 games/pair) ===")
    print(f"=== Output: {base_dir} ===")
    print("=" * 60)

    start_all = time.perf_counter()
    completed = 0

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
                    print(f"[{completed}/{total_pairs}] FAILED {ret_mode} d={ret_d} i={ret_i}: {msg}")
            except Exception as exc:
                print(f"[{completed}/{total_pairs}] ERROR {mode} d={d} i={i}: {exc}")

    total_mins = (time.perf_counter() - start_all) / 60.0
    print("=" * 60)
    print(f"=== ALL PAIRS COMPLETED in {total_mins:.2f} minutes ===")

    master_csv = aggregate_master_standings(base_dir, pairs)
    print(f"=== Master standings written to: {master_csv} ===")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
