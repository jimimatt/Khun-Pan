import numpy as np
import pytest

from khun_pan.khunpan import Board, ShiftDirection


@pytest.mark.parametrize(
    "adjacent_spaces, shiftdirection, aligned_spaces",
    [
        ([(2, 1), (2, 2)], ShiftDirection.LEFT, [(2, 2), (2, 1)]),
        ([(2, 2), (2, 1)], ShiftDirection.LEFT, [(2, 2), (2, 1)]),
        ([(2, 1), (2, 2)], ShiftDirection.RIGHT, [(2, 1), (2, 2)]),
        ([(2, 2), (2, 1)], ShiftDirection.RIGHT, [(2, 1), (2, 2)]),
        ([(2, 2), (3, 2)], ShiftDirection.UP, [(3, 2), (2, 2)]),
        ([(3, 2), (2, 2)], ShiftDirection.UP, [(3, 2), (2, 2)]),
        ([(2, 2), (3, 2)], ShiftDirection.DOWN, [(2, 2), (3, 2)]),
        ([(3, 2), (2, 2)], ShiftDirection.DOWN, [(2, 2), (3, 2)]),
    ],
)
def test_align_spaces(
    adjacent_spaces: list[tuple[int, int]],
    shiftdirection: ShiftDirection,
    aligned_spaces: list[tuple[int, int]],
) -> None:
    """Test align_spaces method."""
    assert Board.align_spaces(adjacent_spaces, shiftdirection) == aligned_spaces


@pytest.mark.parametrize("array, shape", [(np.empty((4, 4)), (2, 2)), (np.empty((7, 6)), (5, 4))])
def test_crop_board(array: np.array, shape: tuple[int, int]) -> None:
    """Test crop_board method."""
    assert Board.crop_board(array).shape == shape


def test_shift_horizontal_h() -> None:
    """Test shift_horizontal_h method."""
    from khun_pan.khunpan import classic_board

    board = Board(board=classic_board())
    board.shift_horizontal_h((3, 1), ShiftDirection.LEFT)
    reference_left_shift = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 3, 3, 0, 0, -1],
            [-1, 2, 1, 1, 2, -1],
            [-1, 2, 1, 1, 2, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    assert (board.board == reference_left_shift).all()
