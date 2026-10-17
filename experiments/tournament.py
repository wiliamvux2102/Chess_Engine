"""Paired matches: identical opening positions and swapped colors."""

import csv
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import chess

from chess_system.ai_interface import ChessAI
from chess_system.game_controller import GameController
from experiments.benchmark import MatchResult, run_match, save_match


def summarize(matches: Iterable[MatchResult]) -> dict[str, dict[str, int | float]]:
    standings: dict[str, dict[str, int | float]] = {}
    for match in matches:
        for name, color in ((match.white, "white"), (match.black, "black")):
            row = standings.setdefault(name, dict(
                games=0, wins=0, draws=0, losses=0, unfinished=0, plies=0, seconds=0.0
            ))
            row["games"] += 1
            played = [m for m in match.moves if m.color == color]
            row["plies"] += len(played)
            row["seconds"] += sum(m.seconds for m in played)
            if match.result == "*":
                row["unfinished"] += 1
            elif match.result == "1/2-1/2":
                row["draws"] += 1
            else:
                won = (match.result == "1-0") == (color == "white")
                row["wins" if won else "losses"] += 1
    return standings


def run_tournament(
    factory_a: Callable[[], ChessAI],
    factory_b: Callable[[], ChessAI],
    *,
    name_a: str = "Alpha-Beta",
    name_b: str = "MCTS",
    openings: Iterable[str] = (chess.STARTING_FEN,),
    rounds: int = 1,
    max_plies: int = 300,
    claim_draw: bool = True,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> list[MatchResult]:
    """Create fresh AI instances for every game; rounds repeat each color pair.

    config should record each algorithm's budget and random seed, if applicable.
    Factories configure agents; this runner does not alter their search settings.
    output_dir, when supplied, must be a new directory for this experiment.
    """
    if rounds < 1 or max_plies < 1 or name_a == name_b:
        raise ValueError("Positive rounds/max_plies and distinct agent names required")
    positions = tuple(openings)
    if not positions:
        raise ValueError("At least one opening is required")
    for fen in positions:
        GameController(fen, claim_draw=claim_draw)
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    matches = []
    for opening_index, fen in enumerate(positions, 1):
        for repetition in range(1, rounds + 1):
            for swapped in (False, True):
                a, b = factory_a(), factory_b()
                match = run_match(
                    b if swapped else a, a if swapped else b,
                    white_name=name_b if swapped else name_a,
                    black_name=name_a if swapped else name_b,
                    fen=fen, max_plies=max_plies, claim_draw=claim_draw, config=config,
                )
                matches.append(match)
                if output is not None:
                    save_match(match, output,
                               f"opening_{opening_index}_round_{repetition}_swap_{int(swapped)}")
    if output is not None:
        with (output / "standings.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "agent", "games", "wins", "draws", "losses", "unfinished", "plies", "seconds"
            ])
            writer.writeheader()
            writer.writerows({"agent": name, **row} for name, row in summarize(matches).items())
    return matches
