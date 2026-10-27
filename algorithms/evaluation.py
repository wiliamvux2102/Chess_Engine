
"""Static evaluation for Alpha-Beta Chess AI.

All scores use White's perspective:
    Positive -> White advantage
    Negative -> Black advantage
    Zero     -> Equal according to this heuristic
"""

import chess


# ============================================================
# 1. Constants
# ============================================================

MATE_SCORE = 100_000

PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


# ============================================================
# 2. Piece-Square Tables
#
# Rows: rank 8 -> rank 1
# Columns: file a -> file h
# Tables are defined from White's perspective.
# ============================================================

PAWN_PST = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (50, 50, 50, 50, 50, 50, 50, 50),
    (10, 10, 20, 30, 30, 20, 10, 10),
    (5, 5, 10, 25, 25, 10, 5, 5),
    (0, 0, 0, 20, 20, 0, 0, 0),
    (5, -5, -10, 0, 0, -10, -5, 5),
    (5, 10, 10, -20, -20, 10, 10, 5),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

KNIGHT_PST = (
    (-50, -40, -30, -30, -30, -30, -40, -50),
    (-40, -20, 0, 0, 0, 0, -20, -40),
    (-30, 0, 10, 15, 15, 10, 0, -30),
    (-30, 5, 15, 20, 20, 15, 5, -30),
    (-30, 0, 15, 20, 20, 15, 0, -30),
    (-30, 5, 10, 15, 15, 10, 5, -30),
    (-40, -20, 0, 5, 5, 0, -20, -40),
    (-50, -40, -30, -30, -30, -30, -40, -50),
)

BISHOP_PST = (
    (-20, -10, -10, -10, -10, -10, -10, -20),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-10, 0, 5, 10, 10, 5, 0, -10),
    (-10, 5, 5, 10, 10, 5, 5, -10),
    (-10, 0, 10, 10, 10, 10, 0, -10),
    (-10, 10, 10, 10, 10, 10, 10, -10),
    (-10, 5, 0, 0, 0, 0, 5, -10),
    (-20, -10, -10, -10, -10, -10, -10, -20),
)

ROOK_PST = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (5, 10, 10, 10, 10, 10, 10, 5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (0, 0, 0, 5, 5, 0, 0, 0),
)

QUEEN_PST = (
    (-20, -10, -10, -5, -5, -10, -10, -20),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-10, 0, 5, 5, 5, 5, 0, -10),
    (-5, 0, 5, 5, 5, 5, 0, -5),
    (0, 0, 5, 5, 5, 5, 0, -5),
    (-10, 5, 5, 5, 5, 5, 0, -10),
    (-10, 0, 5, 0, 0, 0, 0, -10),
    (-20, -10, -10, -5, -5, -10, -10, -20),
)

KING_MIDDLEGAME_PST = (
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-20, -30, -30, -40, -40, -30, -30, -20),
    (-10, -20, -20, -20, -20, -20, -20, -10),
    (20, 20, 0, 0, 0, 0, 20, 20),
    (20, 30, 10, 0, 0, 10, 30, 20),
)

KING_ENDGAME_PST = (
    (-50, -40, -30, -20, -20, -30, -40, -50),
    (-30, -20, -10, 0, 0, -10, -20, -30),
    (-30, -10, 20, 30, 30, 20, -10, -30),
    (-30, -10, 30, 40, 40, 30, -10, -30),
    (-30, -10, 30, 40, 40, 30, -10, -30),
    (-30, -10, 20, 30, 30, 20, -10, -30),
    (-30, -30, 0, 0, 0, 0, -30, -30),
    (-50, -30, -30, -30, -30, -30, -30, -50),
)

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}


# ============================================================
# 3. Material Evaluation
# ============================================================

def evaluate_material(board: chess.Board) -> int:
    """Return material advantage from White's perspective."""
    score = 0

    for piece_type, value in PIECE_VALUES.items():
        white_count = len(
            board.pieces(piece_type, chess.WHITE)
        )
        black_count = len(
            board.pieces(piece_type, chess.BLACK)
        )

        score += (white_count - black_count) * value

    return score


# ============================================================
# 4. Endgame Detection
# ============================================================

def is_endgame(board: chess.Board) -> bool:
    """Detect a simplified endgame using remaining material."""
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))

    # No queens: usually treat as endgame.
    if white_queens == 0 and black_queens == 0:
        return True

    # Sum non-pawn material excluding kings.
    non_pawn_material = 0

    for piece_type in (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    ):
        count = len(board.pieces(piece_type, chess.WHITE))
        count += len(board.pieces(piece_type, chess.BLACK))

        non_pawn_material += count * PIECE_VALUES[piece_type]

    return non_pawn_material <= 1300


# ============================================================
# 5. Positional Evaluation
# ============================================================

def evaluate_position(board: chess.Board) -> int:
    """Return positional advantage from White's perspective."""
    score = 0
    endgame = is_endgame(board)

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            table = (
                KING_ENDGAME_PST
                if endgame
                else KING_MIDDLEGAME_PST
            )
        else:
            table = PIECE_SQUARE_TABLES[piece.piece_type]

        # Convert Black square to White-oriented coordinates.
        if piece.color == chess.WHITE:
            table_square = square
            sign = 1
        else:
            table_square = chess.square_mirror(square)
            sign = -1

        row = 7 - chess.square_rank(table_square)
        col = chess.square_file(table_square)

        score += sign * table[row][col]

    return score


# ============================================================
# 6. Terminal Evaluation
# ============================================================

def evaluate_terminal(
    board: chess.Board,
    ply: int = 0,
    *,
    claim_draw: bool = True,
) -> int | None:
    """Return terminal utility, or None if game is ongoing."""
    outcome = board.outcome(claim_draw=claim_draw)

    if outcome is None:
        return None

    if outcome.winner is None:
        return 0

    if outcome.winner == chess.WHITE:
        return MATE_SCORE - ply

    return -MATE_SCORE + ply


# ============================================================
# 7. Combined Evaluation
# ============================================================

def evaluate(
    board: chess.Board,
    ply: int = 0,
    *,
    claim_draw: bool = True,
) -> int:
    """Evaluate the position from White's perspective."""
    terminal_score = evaluate_terminal(
        board,
        ply=ply,
        claim_draw=claim_draw,
    )

    if terminal_score is not None:
        return terminal_score

    material_score = evaluate_material(board)
    positional_score = evaluate_position(board)

    return material_score + positional_score
