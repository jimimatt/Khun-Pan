from collections import namedtuple

import numpy as np
from khunpan import GameBoard

Puzzle = namedtuple("Puzzle", ["board", "optimal_solution"])


def simple_exercise_1() -> Puzzle:
    """Simple exercise 1."""
    ex1 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 2, 2, 0, 2, -1],
            [-1, 2, 2, 0, 2, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ex1, spaces=[(1, 3), (2, 3)])
    return Puzzle(board=board, optimal_solution=6)


def simple_exercise_2() -> Puzzle:
    """Simple exercise 2."""
    ex2 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, 3, 3, 0, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ex2, spaces=[(1, 1), (1, 4)])
    return Puzzle(board=board, optimal_solution=10)


def simple_exercise_3() -> Puzzle:
    """Simple exercise 3."""
    ex3 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 2, 2, 2, 2, -1],
            [-1, 2, 2, 2, 2, -1],
            [-1, 0, 5, 5, 0, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 1, 3, 3, 1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ex3, spaces=[(3, 1), (3, 4)])
    return Puzzle(board=board, optimal_solution=16)


def simple_exercise_4() -> Puzzle:
    """Simple exercise 4."""
    ex4 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, 3, 3, 0, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ex4, spaces=[(1, 1), (1, 4)])
    return Puzzle(board=board, optimal_solution=29)


def simple_exercise_5() -> Puzzle:
    """Simple exercise 5."""
    ex5 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 0, 3, 3, 0, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ex5, spaces=[(3, 1), (3, 4)])
    return Puzzle(board=board, optimal_solution=32)


def simple_exercise_6() -> Puzzle:
    """Simple exercise 6."""
    ex6 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, 1, 1, 0, -1],
            [-1, 1, 3, 3, 1, -1],
            [-1, 3, 3, 2, 2, -1],
            [-1, 5, 5, 2, 2, -1],
            [-1, 5, 5, 3, 3, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ex6, spaces=[(1, 1), (1, 4)])
    return Puzzle(board=board, optimal_solution=35)


def simple_exercise_7() -> Puzzle:
    """Simple exercise 7."""
    ex7 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 1, 3, 3, 1, -1],
            [-1, 1, 0, 0, 1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ex7, spaces=[(5, 2), (5, 3)])
    return Puzzle(board=board, optimal_solution=36)


def simple_exercise_8() -> Puzzle:
    """Simple exercise 8."""
    ex8 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, 0, 5, 5, 0, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 3, 3, 2, -1],
            [-1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ex8, spaces=[(2, 1), (2, 4)])
    return Puzzle(board=board, optimal_solution=42)


def advanced_exercise_1() -> Puzzle:
    """Advanced exercise 1."""
    ad1 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 2, 5, 5, 1, -1],
            [-1, 2, 5, 5, 1, -1],
            [-1, 0, 3, 3, 0, -1],
            [-1, 1, 3, 3, 2, -1],
            [-1, 1, 3, 3, 2, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ad1, spaces=[(3, 1), (3, 4)])
    return Puzzle(board=board, optimal_solution=85)


def advanced_exercise_2() -> Puzzle:
    """Advanced exercise 2."""
    ad2 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, 5, 5, 0, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 2, 1, 1, 2, -1],
            [-1, 2, 3, 3, 2, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ad2, spaces=[(1, 1), (1, 4)])
    return Puzzle(board=board, optimal_solution=93)


def advanced_exercise_3() -> Puzzle:
    """Advanced exercise 3."""
    ad3 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 0, 3, 3, 0, -1],
            [-1, 1, 1, 1, 1, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ad3, spaces=[(3, 1), (3, 4)])
    return Puzzle(board=board, optimal_solution=94)


def advanced_exercise_4() -> Puzzle:
    """Advanced exercise 4."""
    ad4 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, 5, 5, 0, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 2, 3, 3, 2, -1],
            [-1, 2, 1, 1, 2, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ad4, spaces=[(1, 1), (1, 4)])
    return Puzzle(board=board, optimal_solution=97)


def advanced_exercise_5() -> Puzzle:
    """Advanced exercise 5."""
    ad5 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 1, 5, 5, 2, -1],
            [-1, 2, 3, 3, 2, -1],
            [-1, 2, 3, 3, 1, -1],
            [-1, 0, 3, 3, 0, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ad5, spaces=[(5, 1), (5, 4)])
    return Puzzle(board=board, optimal_solution=98)


def advanced_exercise_6() -> Puzzle:
    """Advanced exercise 6."""
    ad6 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 1, 5, 5, 1, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 2, 0, 2, -1],
            [-1, 1, 2, 0, 1, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ad6, spaces=[(3, 3), (4, 3)])
    return Puzzle(board=board, optimal_solution=99)


def advanced_exercise_7() -> Puzzle:
    """Advanced exercise 7."""
    ad7 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 0, 3, 3, 0, -1],
            [-1, 1, 3, 3, 1, -1],
            [-1, 1, 3, 3, 1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ad7, spaces=[(3, 1), (3, 4)])
    return Puzzle(board=board, optimal_solution=101)


def advanced_exercise_8() -> Puzzle:
    """Advanced exercise 8."""
    ad8 = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 2, 5, 5, 2, -1],
            [-1, 0, 3, 3, 0, -1],
            [-1, 3, 3, 3, 3, -1],
            [-1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=int,
    )
    board = GameBoard(board=ad8, spaces=[(3, 1), (3, 4)])
    return Puzzle(board=board, optimal_solution=104)


if __name__ == "__main__":
    from time import time

    from halo import Halo

    from src.khun_pan.khunpan import KhunPanEscape

    simple_exercises = {
        "simple exercise 1": simple_exercise_1(),
        "simple exercise 2": simple_exercise_2(),
        "simple exercise 3": simple_exercise_3(),
        "simple exercise 4": simple_exercise_4(),
        "simple exercise 5": simple_exercise_5(),
        "simple exercise 6": simple_exercise_6(),
        "simple exercise 7": simple_exercise_7(),
        "simple exercise 8": simple_exercise_8(),
    }

    advanced_exercises = {
        "advanced exercise 1": advanced_exercise_1(),
        "advanced exercise 2": advanced_exercise_2(),
        "advanced exercise 3": advanced_exercise_3(),
        "advanced exercise 4": advanced_exercise_4(),
        "advanced exercise 5": advanced_exercise_5(),
        "advanced exercise 6": advanced_exercise_6(),
        "advanced exercise 7": advanced_exercise_7(),
        "advanced exercise 8": advanced_exercise_8(),
    }

    for name, puzzle in advanced_exercises.items():
        print(name)
        game = KhunPanEscape(puzzle.board)
        start = time()
        with Halo(text='Solving', spinner='dots'):
            game.solve()
        end = time()
        print(f"elapsed time: {end - start:f}")
        solution = game.get_solution()
        print(f"Length of solution found:   {len(solution):d}")
        print(f"Length of optimal solution: {puzzle.optimal_solution:d}")
        print()
