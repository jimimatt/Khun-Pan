import pytest
import numpy as np
from src.khun_pan.khunpan import ShiftDirection, Board


@pytest.mark.parametrize("adjacent_spaces, shiftdirection, aligned_spaces", [
    ([(2, 1), (2, 2)], ShiftDirection.LEFT, [(2, 2), (2, 1)]),
    ([(2, 2), (2, 1)], ShiftDirection.LEFT, [(2, 2), (2, 1)]),
    ([(2, 1), (2, 2)], ShiftDirection.RIGHT, [(2, 1), (2, 2)]),
    ([(2, 2), (2, 1)], ShiftDirection.RIGHT, [(2, 1), (2, 2)]),
    ([(2, 2), (3, 2)], ShiftDirection.UP, [(3, 2), (2, 2)]),
    ([(3, 2), (2, 2)], ShiftDirection.UP, [(3, 2), (2, 2)]),
    ([(2, 2), (3, 2)], ShiftDirection.DOWN, [(2, 2), (3, 2)]),
    ([(3, 2), (2, 2)], ShiftDirection.DOWN, [(2, 2), (3, 2)])
])
def test_align_spaces(adjacent_spaces, shiftdirection, aligned_spaces):
    assert Board.align_spaces(adjacent_spaces, shiftdirection) == aligned_spaces


@pytest.mark.parametrize("array, shape", [
    (np.empty((4, 4)), (2, 2)),
    (np.empty((7, 6)), (5, 4))
])
def test_crop_board(array, shape):
    assert Board.crop_board(array).shape == shape


def test_shift_horizontal_h():
    board = Board()
    board.shift_horizontal_h((3, 1), ShiftDirection.LEFT)
    reference_left_shift = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1, 2, 5, 5, 2, -1],
        [-1, 2, 5, 5, 2, -1],
        [-1, 3, 3, 0, 0, -1],
        [-1, 2, 1, 1, 2, -1],
        [-1, 2, 1, 1, 2, -1],
        [-1, -1, -1, -1, -1, -1]
    ], dtype=int)
    assert (board.board == reference_left_shift).all()
