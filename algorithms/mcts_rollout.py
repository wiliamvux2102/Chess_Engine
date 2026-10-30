"""Chess-aware rollout sampling with material and piece-square evaluation.

Contains the evaluation tables, local exchange estimates, phase-based position
scores, cutoff rewards and weighted move sampling used by MCTS.
"""

import math
import random

import chess

# Indexed by python-chess piece type: none, pawn, knight, bishop, rook, queen, king.
MIDGAME_VALUES = (0, 188, 753, 826, 1285, 2513, 0)
ENDGAME_VALUES = (0, 248, 832, 897, 1371, 2650, 0)
MIDGAME_LIMIT = 15258
ENDGAME_LIMIT = 3915
PHASE_MAX = 128

# One common conversion for material, PSQT and exchange estimates:
# an endgame pawn is the reference, with 248 internal units = 100 cp.
CENTIPAWN_SCALE = 100.0 / ENDGAME_VALUES[1]

# PSQT_BONUS[piece_type][relative_rank][folded_file] -> (middlegame, endgame).
# Ranks run from White's first rank to its eighth; Black mirrors the ranks.
# Files A..D are stored; E..H reflect them. The eighth pawn rank is zero;
# legal positions promote a pawn when it reaches that rank.
PSQT_BONUS = (
    (),
    (  # Pawn
        ((0, 0), (0, 0), (0, 0), (0, 0)),
        ((-16, 7), (1, -4), (7, 8), (3, -2)),
        ((-23, -4), (-7, -5), (19, 5), (24, 4)),
        ((-22, 3), (-14, 3), (20, -8), (35, -3)),
        ((-11, 8), (0, 9), (3, 7), (21, -6)),
        ((-11, 8), (-13, -5), (-6, 2), (-2, 4)),
        ((-9, 3), (15, -9), (-8, 1), (-4, 18)),
        ((0, 0), (0, 0), (0, 0), (0, 0)),
    ),
    (  # Knight
        ((-143, -97), (-96, -82), (-80, -46), (-73, -14)),
        ((-83, -69), (-43, -55), (-21, -17), (-10, 9)),
        ((-71, -50), (-22, -39), (0, -8), (9, 28)),
        ((-25, -41), (18, -25), (43, 7), (47, 38)),
        ((-26, -46), (16, -25), (38, 2), (50, 41)),
        ((-11, -55), (37, -38), (56, -8), (71, 27)),
        ((-62, -64), (-17, -50), (5, -24), (14, 13)),
        ((-195, -110), (-66, -90), (-42, -50), (-29, -13)),
    ),
    (  # Bishop
        ((-54, -68), (-23, -40), (-35, -46), (-44, -28)),
        ((-30, -43), (10, -17), (2, -23), (-9, -5)),
        ((-19, -32), (17, -9), (11, -13), (1, 8)),
        ((-21, -36), (18, -13), (11, -15), (0, 7)),
        ((-21, -36), (14, -14), (6, -17), (-1, 3)),
        ((-27, -35), (6, -13), (2, -10), (-8, 1)),
        ((-33, -44), (7, -21), (-4, -22), (-12, -4)),
        ((-45, -65), (-21, -42), (-29, -46), (-39, -27)),
    ),
    (  # Rook
        ((-25, 0), (-16, 0), (-16, 0), (-9, 0)),
        ((-21, 0), (-8, 0), (-3, 0), (0, 0)),
        ((-21, 0), (-9, 0), (-4, 0), (2, 0)),
        ((-22, 0), (-6, 0), (-1, 0), (2, 0)),
        ((-22, 0), (-7, 0), (0, 0), (1, 0)),
        ((-21, 0), (-7, 0), (0, 0), (2, 0)),
        ((-12, 0), (4, 0), (8, 0), (12, 0)),
        ((-23, 0), (-15, 0), (-11, 0), (-5, 0)),
    ),
    (  # Queen
        ((0, -70), (-3, -57), (-4, -41), (-1, -29)),
        ((-4, -58), (6, -30), (9, -21), (8, -4)),
        ((-2, -39), (6, -17), (9, -7), (9, 5)),
        ((-1, -29), (8, -5), (10, 9), (7, 17)),
        ((-3, -27), (9, -5), (8, 10), (7, 23)),
        ((-2, -40), (6, -16), (8, -11), (10, 3)),
        ((-2, -54), (7, -30), (7, -21), (6, -7)),
        ((-1, -75), (-4, -54), (-1, -44), (0, -30)),
    ),
    (  # King
        ((291, 28), (344, 76), (294, 103), (219, 112)),
        ((289, 70), (329, 119), (263, 170), (205, 159)),
        ((226, 109), (271, 164), (202, 195), (136, 191)),
        ((204, 131), (212, 194), (175, 194), (137, 204)),
        ((177, 132), (205, 187), (143, 224), (94, 227)),
        ((147, 118), (188, 178), (113, 199), (70, 197)),
        ((116, 72), (158, 121), (93, 142), (48, 161)),
        ((94, 30), (120, 76), (78, 101), (31, 111)),
    ),
)


# Fixed endgame material values keep a single scale throughout a local exchange.
# Position evaluation separately blends middlegame and endgame material values.
PIECE_VALUES = {piece_type: ENDGAME_VALUES[piece_type] * CENTIPAWN_SCALE
                for piece_type in chess.PIECE_TYPES}


def exchange_gain(board: chess.Board, move: chess.Move) -> float:
    """Estimate the material gain of a move and its possible recaptures.

    Follow legal captures on the destination square for at most eight replies,
    choosing the least valuable attacker, then fold backward allowing either
    side to decline an exchange. Include en passant and promotion gains.
    Legal captures account for pins and king safety; this remains a local
    heuristic, not a complete tactical search. Restore board/history on exit.

    Parameters:
      [board] - Valid chess.Board before the move; restored even on exceptions.
      [move] - Legal move for the side to move; legality is a caller precondition.

    Returns estimated centipawn gain from the moving side's perspective:
    positive for profit, negative for loss, zero for an even exchange. Material
    uses fixed endgame values normalized to a 100-centipawn pawn.

    Throws no explicit exceptions for valid inputs; board errors propagate
    after restoring any moves pushed by this function.

    Example:
        board = chess.Board("4k3/8/4p3/3r4/8/8/3Q4/4K3 w - - 0 1")
        gain = exchange_gain(board, chess.Move.from_uci("d2d5"))
        assert math.isclose(gain, (1371 - 2650) * 100 / 248)
        # Queen takes rook, then a pawn recaptures queen.
    """
    victim = board.piece_type_at(move.to_square)
    captured = (ENDGAME_VALUES[chess.PAWN] if board.is_en_passant(move)
                else ENDGAME_VALUES[victim] if victim is not None else 0)
    promotion = ENDGAME_VALUES[move.promotion] - ENDGAME_VALUES[chess.PAWN] if move.promotion else 0
    gains = [float(captured + promotion)]
    pushed = 0
    try:
        board.push(move)
        pushed += 1
        for _ in range(8):
            captures = list(board.generate_legal_captures(to_mask=chess.BB_SQUARES[move.to_square]))
            if not captures:
                break
            reply = min(captures, key=lambda candidate: (
                ENDGAME_VALUES[board.piece_type_at(candidate.from_square)]
                if board.piece_type_at(candidate.from_square) != chess.KING else math.inf,
                -(ENDGAME_VALUES[candidate.promotion] if candidate.promotion else 0),
            ))
            captured = ENDGAME_VALUES[board.piece_type_at(move.to_square)]
            promotion = ENDGAME_VALUES[reply.promotion] - ENDGAME_VALUES[chess.PAWN] if reply.promotion else 0
            gains.append(captured + promotion - gains[-1])
            board.push(reply)
            pushed += 1
        for index in range(len(gains) - 1, 0, -1):
            gains[index - 1] = -max(-gains[index - 1], gains[index])
        # Convert once so an even exchange cannot become slightly negative due
        # to repeated floating-point subtraction and trigger a rollout penalty.
        return gains[0] * CENTIPAWN_SCALE
    finally:
        for _ in range(pushed):
            board.pop()


def position_score(board: chess.Board) -> float:
    """Evaluate material and piece-square bonuses from White's view.

    Sum White's material and PSQT bonuses, subtract Black's, and interpolate
    middlegame/endgame totals according to the remaining non-pawn material.
    Normalize all terms with one common scale: 248 internal units = 100 cp.
    This estimate includes material and square placement; mobility, pawn
    structure and a full king-safety model are not included. Terminal outcomes
    are handled separately; this is not a calibrated probability.

    Parameters:
      [board] - Valid chess.Board to inspect; it is not modified.

    Returns a centipawn score: positive favors White, negative favors Black.

    Throws no explicit exceptions for a valid board.

    Example:
        assert position_score(chess.Board()) == 0.0
    """
    middlegame, endgame, non_pawn_material = 0, 0, 0
    for square, piece in board.piece_map().items():
        file = chess.square_file(square)
        rank = chess.square_rank(square) if piece.color else 7 - chess.square_rank(square)
        piece_type = piece.piece_type
        mg_bonus, eg_bonus = PSQT_BONUS[piece_type][rank][min(file, 7 - file)]
        sign = 1 if piece.color else -1
        middlegame += sign * (MIDGAME_VALUES[piece_type] + mg_bonus)
        endgame += sign * (ENDGAME_VALUES[piece_type] + eg_bonus)
        if piece_type not in (chess.PAWN, chess.KING):
            non_pawn_material += MIDGAME_VALUES[piece_type]
    material = max(ENDGAME_LIMIT, min(non_pawn_material, MIDGAME_LIMIT))
    phase = (material - ENDGAME_LIMIT) * PHASE_MAX // (MIDGAME_LIMIT - ENDGAME_LIMIT)
    # Blend the two phases smoothly and retain fractional centipawn values.
    return ((middlegame * phase + endgame * (PHASE_MAX - phase))
            / PHASE_MAX * CENTIPAWN_SCALE)


def cutoff_value(board: chess.Board) -> float:
    """Evaluate an unfinished rollout with an immediate-exchange correction.

    Adjust position_score by the best nonnegative exchange available to the
    current player, then apply 0.98 * tanh(score / 600). This reduces horizon
    errors when a capture can be answered immediately. Local exchange trials
    restore the board/history and do not extend the sampled rollout.

    Parameters:
      [board] - Valid, nonterminal chess.Board at the rollout cutoff.

    Returns an estimated White reward in [-0.98, +0.98]. Zero means heuristic
    balance; exact terminal rewards -1, 0 and +1 are handled by the caller.

    Throws no explicit exceptions for valid inputs; evaluation errors propagate.

    Example:
        assert cutoff_value(chess.Board()) == 0.0
    """
    best_capture = max((exchange_gain(board, move) for move in board.generate_legal_captures()),
                       default=0.0)
    score = position_score(board) + (1 if board.turn else -1) * max(0.0, best_capture)
    return 0.98 * math.tanh(score / 600.0)


def rollout_move(
    board: chess.Board, rng: random.Random, moves: list[chess.Move] | None = None,
) -> chess.Move:
    """Choose a candidate move using tactical checks and weighted exploration.

    Take a proven immediate mate first. Otherwise choose uniformly with 10%
    probability or sample weights favoring profitable exchanges, promotions,
    checks and central minor-piece moves. Quiet-move attack penalties remain
    approximate. The helper is shared by rollouts and deeper expansion.

    Parameters:
      [board] - Valid position before the candidate move; board/history remain
          unchanged when the helper returns.
      [rng] - Private random.Random generator controlling the sampling.
      [moves] - Optional nonempty list of legal candidates, such as untried
          moves; None uses all legal moves. Legality is a caller precondition.

    Returns one chess.Move from the supplied or generated candidate list.

    Throws [ValueError] if the candidate list is empty, including a terminal
    position with no legal moves. Board/evaluation errors otherwise propagate.

    Example:
        board = chess.Board()
        candidates = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        move = rollout_move(board, random.Random(42), candidates)
        assert move in candidates
    """
    moves = list(board.legal_moves) if moves is None else list(moves)
    if not moves:
        raise ValueError("At least one candidate move is required")
    # A verified immediate mate takes priority over both sampling branches.
    checks = set()
    for move in moves:
        if not board.gives_check(move):
            continue
        checks.add(move)
        board.push(move)
        try:
            if board.is_checkmate():
                return move
        finally:
            board.pop()
    # Uniform exploration keeps low-weight moves reachable.
    if rng.random() < 0.1:
        return rng.choice(moves)
    weights = []
    for move in moves:
        moving = board.piece_type_at(move.from_square)
        tactical = board.is_capture(move) or move.promotion
        # Captures/promotions use a local exchange estimate, not a full search.
        gain = exchange_gain(board, move) if tactical else 0
        attacked = board.is_attacked_by(not board.turn, move.to_square)
        if attacked and not tactical:
            # Approximate risk for quiet moves using attacks before the move.
            gain -= PIECE_VALUES.get(moving, 0)
        weight = 1.0 + max(0, gain) / 30.0
        if gain < 0 and moving != chess.KING:
            weight *= 0.1
        if move in checks:
            weight *= 1.5
        if moving in (chess.KNIGHT, chess.BISHOP):
            file, rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
            weight *= 1 + (min(file, 7 - file) + min(rank, 7 - rank)) / 6
        weights.append(weight)
    # Sample proportionally to weights; do not always take the largest weight.
    return rng.choices(moves, weights=weights, k=1)[0]
