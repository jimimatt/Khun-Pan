# Khun-Pan

This project is a solver of the traditional sliding puzzle game Khun Pan.
The game is named after a legendary Thai warrior and features a simple yet captivating gameplay where the objective is to slide the warrior piece to the exit.

The solver processes a game state and adds each (new / unseen) possible move to a queue of game states beginning with the starting position.
This breadth-first search algorithm finds the shortest solution to the puzzle.
Each processed game state is encoded in an integer and stored in an ordered list to prevent revisiting the same state.

The solver resides in the `khunpan.py`, running this script will print the shortest solution of the classic problem to the console.

`puzzles.py` contains a few example puzzles to test the solver.
