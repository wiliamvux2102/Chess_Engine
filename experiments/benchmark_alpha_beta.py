"""Compare Alpha-Beta with and without move ordering."""

from statistics import mean
from time import perf_counter
from unittest.mock import patch

import chess

from algorithms.alpha_beta import AlphaBetaAI


POSITIONS = {
    "Starting position": chess.STARTING_FEN,
    "After 1.e4 e5": (
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/"
        "PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    ),
    "Castling available": (
        "r3k2r/8/8/8/8/8/8/"
        "R3K2R w KQkq - 0 1"
    ),
}


def run_search(
    fen: str,
    depth: int,
    *,
    use_ordering: bool,
) -> dict:
    board = chess.Board(fen)
    ai = AlphaBetaAI(depth=depth)

    def original_order(b: chess.Board):
        return list(b.legal_moves)

    if use_ordering:
        start = perf_counter()
        move = ai.choose_move(board)
        elapsed = perf_counter() - start
    else:
        with patch(
            "algorithms.alpha_beta.order_moves",
            original_order,
        ):
            start = perf_counter()
            move = ai.choose_move(board)
            elapsed = perf_counter() - start

    return {
        "move": move.uci(),
        "seconds": elapsed,
        "nodes": ai.last_search_stats["nodes"],
        "cutoffs": ai.last_search_stats["cutoffs"],
    }


def main():
    depth = 4
    repetitions = 10

    for name, fen in POSITIONS.items():
        print(f"\nPosition: {name}")

        for use_ordering in (False, True):
            results = [
                run_search(
                    fen,
                    depth,
                    use_ordering=use_ordering,
                )
                for _ in range(repetitions)
            ]

            label = (
                "Ordered"
                if use_ordering
                else "Unordered"
            )

            print(
                f"{label}: "
                f"mean_time={mean(r['seconds'] for r in results):.4f}s, "
                f"nodes={mean(r['nodes'] for r in results):.1f}, "
                f"cutoffs={mean(r['cutoffs'] for r in results):.1f}, "
                f"move={results[0]['move']}"
            )


if __name__ == "__main__":
    main()
