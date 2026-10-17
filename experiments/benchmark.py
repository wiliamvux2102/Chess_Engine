"""Run one game and persist per-move timing, metrics, PGN, and results."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import chess
import chess.pgn

from chess_system.ai_interface import ChessAI
from chess_system.game_controller import GameController


@dataclass(frozen=True)
class MoveRecord:
    ply: int
    color: str
    agent: str
    move: str
    seconds: float
    depth: int | None = None
    nodes: int | None = None
    iterations: int | None = None
    rollouts: int | None = None


@dataclass
class MatchResult:
    white: str
    black: str
    initial_fen: str
    result: str
    termination: str
    moves: list[MoveRecord]
    pgn: str
    config: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "white": self.white, "black": self.black,
            "initial_fen": self.initial_fen, "result": self.result,
            "termination": self.termination, "plies": len(self.moves),
            "white_seconds": sum(m.seconds for m in self.moves if m.color == "white"),
            "black_seconds": sum(m.seconds for m in self.moves if m.color == "black"),
            "config": self.config,
        }


def run_match(
    white: ChessAI,
    black: ChessAI,
    *,
    white_name: str = "white",
    black_name: str = "black",
    fen: str = chess.STARTING_FEN,
    max_plies: int = 300,
    claim_draw: bool = True,
    config: dict[str, Any] | None = None,
) -> MatchResult:
    """Measure choose_move wall time only; reject broken AI contracts.

    Agents enforce their own search budgets. max_plies limits game length,
    not thinking time. A game stopped by this limit remains unfinished (*).
    Exceptions propagate instead of silently counting an AI bug as a loss.
    """
    if max_plies < 1:
        raise ValueError("max_plies must be positive")
    controller = GameController(fen, claim_draw=claim_draw)
    game = chess.pgn.Game.from_board(controller.board)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    node = game
    records = []
    for ply in range(1, max_plies + 1):
        if controller.is_terminal():
            break
        is_white = controller.turn == chess.WHITE
        agent = white if is_white else black
        name = white_name if is_white else black_name
        snapshot = controller.board
        before = (snapshot.fen(), tuple(snapshot.move_stack))
        start = perf_counter()
        move = agent.choose_move(snapshot)
        elapsed = perf_counter() - start
        if (snapshot.fen(), tuple(snapshot.move_stack)) != before:
            raise ValueError(f"{name} changed its input board")
        if not isinstance(move, chess.Move):
            raise TypeError(f"{name} must return chess.Move")
        controller.push(move)
        stats = getattr(agent, "last_search_stats", {})
        if not isinstance(stats, dict):
            raise TypeError("last_search_stats must be a dict")
        metrics = {}
        for key in ("depth", "nodes", "iterations", "rollouts"):
            value = stats.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Invalid search metric: {key}={value!r}")
            metrics[key] = value
        records.append(MoveRecord(
            ply, "white" if is_white else "black", name, move.uci(), elapsed, **metrics
        ))
        node = node.add_variation(move)
    outcome = controller.result()
    termination = outcome.termination or "MOVE_LIMIT"
    game.headers["Result"] = outcome.result
    game.headers["Termination"] = termination
    settings = dict(config or {})
    settings.update(max_plies=max_plies, claim_draw=claim_draw)
    return MatchResult(white_name, black_name, fen, outcome.result,
                       termination, records, str(game), settings)


def save_match(match: MatchResult, output_dir: str | Path, match_id: str) -> None:
    """Use a unique match_id; refuse to overwrite existing experiment files."""
    if not match_id or Path(match_id).name != match_id or match_id in (".", ".."):
        raise ValueError("match_id must be a plain filename")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{match_id}{suffix}" for suffix in (".json", ".csv", ".pgn")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Match already exists: {match_id}")
    with paths[0].open("x", encoding="utf-8") as stream:
        json.dump({**match.summary(), "moves": [asdict(m) for m in match.moves]},
                  stream, indent=2, ensure_ascii=False)
    with paths[1].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(MoveRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(m) for m in match.moves)
    paths[2].write_text(match.pgn + "\n", encoding="utf-8")
