"""8x8 Chessboard Canvas coordinate mapping and rendering."""

from __future__ import annotations

import tkinter as tk

import chess

from gui.theme import (
    BOARD_PX,
    CHECK_SQ,
    COORD_FONT,
    DARK_SQ,
    LAST_MOVE_DARK,
    LAST_MOVE_LIGHT,
    LIGHT_SQ,
    MARGIN,
    PIECE_FONT,
    PROMO_LABEL_FONT,
    PROMOTION_PIECES,
    SELECTED_SQ,
    SQ_SIZE,
    UNICODE_PIECES,
)


def square_to_grid(square: chess.Square, flipped: bool = False) -> tuple[int, int]:
    """Convert chess.Square to (col, row) canvas grid coordinates (0..7)."""
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    if flipped:
        return 7 - file_idx, rank_idx
    return file_idx, 7 - rank_idx


def pixel_to_square(x: int, y: int, flipped: bool = False) -> chess.Square | None:
    """Convert canvas pixel (x, y) to a chess.Square, or None if outside the 8x8 grid."""
    col = (x - MARGIN) // SQ_SIZE
    row = (y - MARGIN) // SQ_SIZE
    if not (0 <= col < 8 and 0 <= row < 8):
        return None
    file_idx = 7 - col if flipped else col
    rank_idx = row if flipped else 7 - row
    return chess.square(file_idx, rank_idx)


def get_promotion_targets(
    pending_promotion: tuple[chess.Square, chess.Square, list[chess.Move]] | None,
    flipped: bool = False,
) -> list[tuple[chess.Square, int, int, chess.PieceType, str]]:
    """Return promotion option squares and grid coordinates for the popup."""
    if pending_promotion is None:
        return []
    _, to_sq, _ = pending_promotion
    col, start_row = square_to_grid(to_sq, flipped)
    step = 1 if start_row < 4 else -1
    items: list[tuple[chess.Square, int, int, chess.PieceType, str]] = []
    for idx, (label, pt) in enumerate(PROMOTION_PIECES):
        row = start_row + idx * step
        if 0 <= row < 8:
            f_idx = 7 - col if flipped else col
            r_idx = row if flipped else 7 - row
            opt_sq = chess.square(f_idx, r_idx)
            items.append((opt_sq, col, row, pt, label))
    return items


def draw_board(
    canvas: tk.Canvas,
    board: chess.Board,
    *,
    flipped: bool = False,
    selected_square: chess.Square | None = None,
    legal_targets: dict[chess.Square, list[chess.Move]] | None = None,
    last_move: chess.Move | None = None,
    pending_promotion: tuple[chess.Square, chess.Square, list[chess.Move]] | None = None,
    hover_promo_sq: chess.Square | None = None,
) -> None:
    """Render the full 8x8 board, piece glyphs, coordinates, and promotion dialog on canvas."""
    canvas.delete("all")
    targets = legal_targets or {}
    king_in_check_sq = board.king(board.turn) if board.is_check() else None
    last_squares = (
        {last_move.from_square, last_move.to_square} if last_move is not None else set()
    )

    # 1. Draw Squares & Highlights
    for square in chess.SQUARES:
        col, row = square_to_grid(square, flipped)
        x1 = MARGIN + col * SQ_SIZE
        y1 = MARGIN + row * SQ_SIZE
        x2 = x1 + SQ_SIZE
        y2 = y1 + SQ_SIZE

        is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        fill = LIGHT_SQ if is_light else DARK_SQ
        if square in last_squares:
            fill = LAST_MOVE_LIGHT if is_light else LAST_MOVE_DARK
        if square == selected_square:
            fill = SELECTED_SQ
        if square == king_in_check_sq:
            fill = CHECK_SQ

        canvas.create_rectangle(
            x1, y1, x2, y2, fill=fill, outline="", tags=f"sq_{square}"
        )

        # Legal target indicator
        if square in targets:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if board.piece_at(square) is not None or (
                selected_square is not None
                and board.is_en_passant(targets[square][0])
            ):
                canvas.create_oval(
                    x1 + 4,
                    y1 + 4,
                    x2 - 4,
                    y2 - 4,
                    outline="#3a6b35",
                    width=4,
                    tags="legal_hint",
                )
            else:
                r = 9
                canvas.create_oval(
                    cx - r,
                    cy - r,
                    cx + r,
                    cy + r,
                    fill="#4a7846",
                    outline="",
                    tags="legal_hint",
                )

        # Piece glyph (solid silhouette with shadow outline)
        piece = board.piece_at(square)
        if piece is not None:
            glyph = UNICODE_PIECES[(piece.color, piece.piece_type)]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if piece.color == chess.WHITE:
                # Seal internal slits and create a crisp dark outline around the solid glyph
                for dx, dy in (
                    (-2, 0),
                    (2, 0),
                    (0, -2),
                    (0, 2),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                    (1, 1),
                ):
                    canvas.create_text(
                        cx + dx,
                        cy + dy,
                        text=glyph,
                        fill="#18191c",
                        font=PIECE_FONT,
                    )
                canvas.create_text(
                    cx,
                    cy,
                    text=glyph,
                    fill="#ffffff",
                    font=PIECE_FONT,
                    tags="piece",
                )
            else:
                for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                    canvas.create_text(
                        cx + dx,
                        cy + dy,
                        text=glyph,
                        fill="#d4d4d8",
                        font=PIECE_FONT,
                    )
                canvas.create_text(
                    cx,
                    cy,
                    text=glyph,
                    fill="#18191c",
                    font=PIECE_FONT,
                    tags="piece",
                )

    # 2. Outer border around the 8x8 playing grid
    grid_end = MARGIN + 8 * SQ_SIZE
    canvas.create_rectangle(
        MARGIN,
        MARGIN,
        grid_end,
        grid_end,
        outline="#3f4147",
        width=2,
    )

    # 3. File and rank coordinates
    for idx in range(8):
        file_char = chess.FILE_NAMES[7 - idx if flipped else idx]
        rank_char = chess.RANK_NAMES[idx if flipped else 7 - idx]
        fx = MARGIN + idx * SQ_SIZE + SQ_SIZE // 2
        ry = MARGIN + idx * SQ_SIZE + SQ_SIZE // 2
        canvas.create_text(
            fx,
            BOARD_PX - MARGIN // 2,
            text=file_char,
            fill="#b5bac1",
            font=COORD_FONT,
        )
        canvas.create_text(
            MARGIN // 2,
            ry,
            text=rank_char,
            fill="#b5bac1",
            font=COORD_FONT,
        )

    # 4. Chess.com-style in-board promotion column overlay
    if pending_promotion is not None:
        promo_targets = get_promotion_targets(pending_promotion, flipped)
        if promo_targets:
            cols = [c for _, c, _, _, _ in promo_targets]
            rows = [r for _, _, r, _, _ in promo_targets]
            px1 = MARGIN + min(cols) * SQ_SIZE
            py1 = MARGIN + min(rows) * SQ_SIZE
            px2 = MARGIN + (max(cols) + 1) * SQ_SIZE
            py2 = MARGIN + (max(rows) + 1) * SQ_SIZE

            # Drop shadow + elevated card container
            canvas.create_rectangle(
                px1 + 3,
                py1 + 3,
                px2 + 3,
                py2 + 3,
                fill="#0d0e10",
                outline="",
                tags="promo_overlay",
            )
            canvas.create_rectangle(
                px1,
                py1,
                px2,
                py2,
                fill="#f8f9fa",
                outline="#81b64c",
                width=3,
                tags="promo_overlay",
            )

            promo_color = board.turn
            for opt_sq, col, row, pt, label in promo_targets:
                sx1 = MARGIN + col * SQ_SIZE
                sy1 = MARGIN + row * SQ_SIZE
                sx2 = sx1 + SQ_SIZE
                sy2 = sy1 + SQ_SIZE
                cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                hovered = opt_sq == hover_promo_sq

                bg_fill = "#e2f0cb" if hovered else "#f8f9fa"
                canvas.create_rectangle(
                    sx1 + 2,
                    sy1 + 2,
                    sx2 - 2,
                    sy2 - 2,
                    fill=bg_fill,
                    outline="#d0d4dc",
                    width=1,
                    tags="promo_overlay",
                )
                disc_r = SQ_SIZE // 2 - 5
                disc_fill = "#ffffff" if hovered else "#e9ecef"
                disc_outline = "#81b64c" if hovered else "#ced4da"
                canvas.create_oval(
                    cx - disc_r,
                    cy - disc_r,
                    cx + disc_r,
                    cy + disc_r,
                    fill=disc_fill,
                    outline=disc_outline,
                    width=2 if hovered else 1,
                    tags="promo_overlay",
                )

                glyph = UNICODE_PIECES[(promo_color, pt)]
                if promo_color == chess.WHITE:
                    for dx, dy in (
                        (-2, 0),
                        (2, 0),
                        (0, -2),
                        (0, 2),
                        (-1, -1),
                        (1, -1),
                        (-1, 1),
                        (1, 1),
                    ):
                        canvas.create_text(
                            cx + dx,
                            cy + dy,
                            text=glyph,
                            fill="#18191c",
                            font=PIECE_FONT,
                            tags="promo_overlay",
                        )
                    canvas.create_text(
                        cx,
                        cy,
                        text=glyph,
                        fill="#ffffff",
                        font=PIECE_FONT,
                        tags="promo_overlay",
                    )
                else:
                    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                        canvas.create_text(
                            cx + dx,
                            cy + dy,
                            text=glyph,
                            fill="#ffffff",
                            font=PIECE_FONT,
                            tags="promo_overlay",
                        )
                    canvas.create_text(
                        cx,
                        cy,
                        text=glyph,
                        fill="#18191c",
                        font=PIECE_FONT,
                        tags="promo_overlay",
                    )

                canvas.create_text(
                    sx2 - 9,
                    sy1 + 9,
                    text=label[0],
                    fill="#495057",
                    font=PROMO_LABEL_FONT,
                    tags="promo_overlay",
                )
