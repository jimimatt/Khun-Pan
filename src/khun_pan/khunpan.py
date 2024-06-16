import bisect
from collections import deque, namedtuple
from copy import deepcopy
from enum import Enum, IntEnum

import numpy as np


class Tile(IntEnum):
    """Enum for the different tiles on the board."""

    KHUNPAN = 5
    VERTICAL = 2
    HORIZONTAL = 3
    SINGLE = 1
    EMPTY = 0
    BORDER = -1


class ShiftDirection(Enum):
    """Enum for the different shift directions."""

    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Adjacency(Enum):
    """Enum for the different adjacency types."""

    NON = 0
    VERTICAL = 1
    HORIZONTAL = 2


GameBoard = namedtuple("GameBoard", ["board", "spaces"])


def classic_board() -> GameBoard:
    """Create the classic board for the Khun Pan puzzle."""
    board = np.zeros([7, 6], dtype=int)
    board[:, 0] = Tile.BORDER
    board[:, -1] = Tile.BORDER
    board[0, :] = Tile.BORDER
    board[-1, :] = Tile.BORDER
    board[1:3, 2:4] = Tile.KHUNPAN
    board[1:3, 1] = Tile.VERTICAL
    board[1:3, 4] = Tile.VERTICAL
    board[4:6, 1] = Tile.VERTICAL
    board[4:6, 4] = Tile.VERTICAL
    board[3, 2:4] = Tile.HORIZONTAL
    board[4, 2] = Tile.SINGLE
    board[4, 3] = Tile.SINGLE
    board[5, 2] = Tile.SINGLE
    board[5, 3] = Tile.SINGLE
    return GameBoard(board, [(3, 1), (3, 4)])


class Board:
    """Class for the Khun Pan board."""

    def __init__(self, board: GameBoard = classic_board()) -> None:
        """Initialize the board."""
        board = deepcopy(board)
        self.board = board.board
        self.spaces = board.spaces

    def __str__(self) -> str:
        string = ""
        for row in Board.crop_board(self.board):
            string += " ".join([str(x) for x in row]) + "\n"
        return string

    def get_tile(self, coord: tuple[int, int]) -> int:
        """Get the tile at the given coordinates."""
        return self.board[coord]

    def check_win_condition(self) -> bool:
        """Check if the win condition is met."""
        return np.all(self.board[4:6, 2:4] == Tile.KHUNPAN)

    @staticmethod
    def get_north_coord(coord: tuple[int, int]) -> tuple[int, int]:
        """Get the coordinates of the tile north of the given coordinates."""
        return coord[0] - 1, coord[1]

    @staticmethod
    def get_south_coord(coord: tuple[int, int]) -> tuple[int, int]:
        """Get the coordinates of the tile south of the given coordinates."""
        return coord[0] + 1, coord[1]

    @staticmethod
    def get_east_coord(coord: tuple[int, int]) -> tuple[int, int]:
        """Get the coordinates of the tile east of the given coordinates."""
        return coord[0], coord[1] + 1

    @staticmethod
    def get_west_coord(coord: tuple[int, int]) -> tuple[int, int]:
        """Get the coordinates of the tile west of the given coordinates."""
        return coord[0], coord[1] - 1

    @staticmethod
    def align_spaces(
        spaces: list[tuple[int, int]], direction: ShiftDirection
    ) -> list[tuple[int, int]]:
        """Align the spaces based on the shift direction."""

        def swap_spaces(space_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
            space_list[0], space_list[1] = space_list[1], space_list[0]
            return space_list

        space_a, space_b = spaces
        if (
            (direction == ShiftDirection.UP and space_a[0] < space_b[0])
            or (direction == ShiftDirection.DOWN and space_a[0] > space_b[0])
            or (direction == ShiftDirection.LEFT and space_a[1] < space_b[1])
            or (direction == ShiftDirection.RIGHT and space_a[1] > space_b[1])
        ):
            spaces = swap_spaces(spaces)
        return spaces

    def shift_single(
        self,
        space: tuple[int, int],
        direction: ShiftDirection,
        tile: Tile = Tile.SINGLE,
    ) -> None:
        """Shift a single tile in the given direction."""
        self.spaces.remove(space)
        if direction == ShiftDirection.UP:
            pos = (space[0] + 1, space[1])
        elif direction == ShiftDirection.DOWN:
            pos = (space[0] - 1, space[1])
        elif direction == ShiftDirection.LEFT:
            pos = (space[0], space[1] + 1)
        elif direction == ShiftDirection.RIGHT:
            pos = (space[0], space[1] - 1)
        self.board[pos] = Tile.EMPTY
        self.board[space] = tile
        self.spaces.append(pos)

    def shift_single_twice(
        self, spaces: list[tuple[int, int], tuple[int, int]], direction: ShiftDirection
    ) -> None:
        """Shift two single tiles in the given direction."""
        spaces = Board.align_spaces(spaces, direction)
        for space in spaces:
            self.shift_single(space, direction)

    def shift_vertical_v(
        self,
        space: tuple[int, int],
        direction: ShiftDirection,
        tile: Tile = Tile.VERTICAL,
    ) -> None:
        """Shift a vertical tile in the given direction."""
        self.spaces.remove(space)
        if direction == ShiftDirection.UP:
            new_space = (space[0] + 2, space[1])
            pos = (slice(space[0], space[0] + 2), space[1])
        elif direction == ShiftDirection.DOWN:
            new_space = (space[0] - 2, space[1])
            pos = (slice(space[0] - 1, space[0] + 1), space[1])
        self.board[new_space] = Tile.EMPTY
        self.board[pos] = tile
        self.spaces.append(new_space)

    def shift_vertical_v_twice(
        self, spaces: list[tuple[int, int], tuple[int, int]], direction: ShiftDirection
    ) -> None:
        """Shift two vertical tiles in the given direction."""
        spaces = Board.align_spaces(spaces, direction)
        for space in spaces:
            self.shift_vertical_v(space, direction)

    def shift_horizontal_h(
        self,
        space: tuple[int, int],
        direction: ShiftDirection,
        tile: Tile = Tile.HORIZONTAL,
    ) -> None:
        """Shift a horizontal tile in the given direction."""
        self.spaces.remove(space)
        if direction == ShiftDirection.LEFT:
            new_space = (space[0], space[1] + 2)
            pos = (space[0], slice(space[1], space[1] + 2))
        elif direction == ShiftDirection.RIGHT:
            new_space = (space[0], space[1] - 2)
            pos = (space[0], slice(space[1] - 1, space[1] + 1))
        self.board[new_space] = Tile.EMPTY
        self.board[pos] = tile
        self.spaces.append(new_space)

    def shift_horizontal_h_twice(
        self, spaces: list[tuple[int, int], tuple[int, int]], direction: ShiftDirection
    ) -> None:
        """Shift two horizontal tiles in the given direction."""
        spaces = Board.align_spaces(spaces, direction)
        for space in spaces:
            self.shift_horizontal_h(space, direction)

    def shift_vertical_h(
        self, spaces: list[tuple[int, int], tuple[int, int]], direction: ShiftDirection
    ) -> None:
        """Shift two vertical tiles in the given direction."""
        for space in spaces:
            self.shift_single(space, direction, Tile.VERTICAL)

    def shift_horizontal_v(
        self, spaces: list[tuple[int, int], tuple[int, int]], direction: ShiftDirection
    ) -> None:
        """Shift two horizontal tiles in the given direction."""
        for space in spaces:
            self.shift_single(space, direction, Tile.HORIZONTAL)

    def shift_khunpan(
        self, spaces: list[tuple[int, int], tuple[int, int]], direction: ShiftDirection
    ) -> None:
        """Shift two khunpan tiles in the given direction."""
        for space in spaces:
            if direction == ShiftDirection.UP or direction == ShiftDirection.DOWN:
                self.shift_vertical_v(space, direction, Tile.KHUNPAN)
            elif direction == ShiftDirection.LEFT or direction == ShiftDirection.RIGHT:
                self.shift_horizontal_h(space, direction, Tile.KHUNPAN)

    def encode(self) -> np.uint64:
        """Encode the board to a 64-bit integer."""
        board = Board.crop_board(self.board)
        ser = np.uint64(0)
        i = 0
        for row in board:
            for tile in row:
                tile_pos = np.left_shift(tile, i, dtype=np.uint64)
                ser += np.uint64(tile_pos)
                i += 3
        return ser

    @staticmethod
    def decode(ser_board: np.uint64, shape: list[int, int] = (5, 4)) -> GameBoard:
        """Decode a 64-bit integer to a board."""
        board = np.empty(shape, dtype=int)
        spaces = []
        k = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                tile = int(np.bitwise_and(ser_board, 7, dtype=np.uint64))
                board[i, j] = tile
                if tile == 0:
                    spaces.append((i + 1, j + 1))
                ser_board = np.right_shift(ser_board, 3, dtype=np.uint64)
                k += 3
        framed_board = Board.frame_board(board)
        return GameBoard(framed_board, spaces)

    @staticmethod
    def crop_board(board: np.ndarray) -> np.ndarray:
        """Crop the board to remove the border."""
        return board[1:-1, 1:-1]

    @staticmethod
    def frame_board(board: np.ndarray) -> np.ndarray:
        """Frame the board with a border."""
        framed_board = np.full(
            [board.shape[0] + 2, board.shape[1] + 2], fill_value=Tile.BORDER, dtype=int
        )
        framed_board[1:-1, 1:-1] = board
        return framed_board


class KhunPanBoard:
    """Class for the Khun Pan board with additional functionality."""

    instance_count = 0

    def __init__(self, board: Board = Board(), predecessor: int = 0) -> None:
        """Initialize the board."""
        KhunPanBoard.instance_count += 1
        self.id = self.instance_count
        self.board = board
        self.predecessor = predecessor

    def is_won(self) -> bool:
        """Check if the win condition is met."""
        return self.board.check_win_condition()

    def get_move(
        self,
        boards: list[Board],
        space: tuple[int, int],
        tile: Tile,
        direction: ShiftDirection,
    ) -> list[Board]:
        """Get the board after a move in the given direction."""
        board = deepcopy(self.board)
        if tile == Tile.SINGLE:
            board.shift_single(space, direction)
            boards.append(board)
        elif tile == Tile.VERTICAL and (
            direction == ShiftDirection.UP or direction == ShiftDirection.DOWN
        ):
            board.shift_vertical_v(space, direction)
            boards.append(board)
        elif tile == Tile.HORIZONTAL and (
            direction == ShiftDirection.LEFT or direction == ShiftDirection.RIGHT
        ):
            board.shift_horizontal_h(space, direction)
            boards.append(board)
        return boards

    def get_double_move(
        self, boards: list[Board], tile: Tile, direction: ShiftDirection
    ) -> list[Board]:
        """Get the board after a double move in the given direction."""
        board = deepcopy(self.board)
        if tile == Tile.SINGLE:
            board.shift_single_twice(self.board.spaces, direction)
            boards.append(board)
        elif tile == Tile.VERTICAL and (
            direction == ShiftDirection.UP or direction == ShiftDirection.DOWN
        ):
            board.shift_vertical_v_twice(self.board.spaces, direction)
            boards.append(board)
        elif tile == Tile.HORIZONTAL and (
            direction == ShiftDirection.LEFT or direction == ShiftDirection.RIGHT
        ):
            board.shift_horizontal_h_twice(self.board.spaces, direction)
            boards.append(board)
        return boards

    def get_adjacent_moves_vertical(self, boards: list[Board]) -> list[Board]:
        """Get the boards after adjacent moves in the vertical direction."""
        spaces = Board.align_spaces(self.board.spaces, ShiftDirection.DOWN)
        upperwest = Board.get_west_coord(spaces[0])
        lowerwest = Board.get_west_coord(spaces[1])
        northwest = Board.get_north_coord(upperwest)
        southwest = Board.get_south_coord(lowerwest)
        west_a_tile = self.board.get_tile(upperwest)
        west_b_tile = self.board.get_tile(lowerwest)
        nw_tile = self.board.get_tile(northwest)
        sw_tile = self.board.get_tile(southwest)
        if (
            west_a_tile == Tile.VERTICAL
            and west_b_tile == Tile.VERTICAL
            and not (nw_tile == Tile.VERTICAL and sw_tile == Tile.VERTICAL)
        ):
            board = deepcopy(self.board)
            board.shift_vertical_h(
                spaces=self.board.spaces, direction=ShiftDirection.RIGHT
            )
            boards.append(board)
        elif west_a_tile == Tile.KHUNPAN and west_b_tile == Tile.KHUNPAN:
            board = deepcopy(self.board)
            board.shift_khunpan(
                spaces=self.board.spaces, direction=ShiftDirection.RIGHT
            )
            boards.append(board)
        if west_a_tile == Tile.SINGLE:
            board = deepcopy(self.board)
            board.shift_single(spaces[0], ShiftDirection.RIGHT)
            board.shift_single(spaces[1], ShiftDirection.DOWN)
            boards.append(board)
        if west_b_tile == Tile.SINGLE:
            board = deepcopy(self.board)
            board.shift_single(spaces[1], ShiftDirection.RIGHT)
            board.shift_single(spaces[0], ShiftDirection.UP)
            boards.append(board)
        uppereast = Board.get_east_coord(spaces[0])
        lowereast = Board.get_east_coord(spaces[1])
        northeast = Board.get_north_coord(uppereast)
        southeast = Board.get_south_coord(lowereast)
        east_a_tile = self.board.get_tile(uppereast)
        east_b_tile = self.board.get_tile(lowereast)
        ne_tile = self.board.get_tile(northeast)
        se_tile = self.board.get_tile(southeast)
        if (
            east_a_tile == Tile.VERTICAL
            and east_b_tile == Tile.VERTICAL
            and not (ne_tile == Tile.VERTICAL and se_tile == Tile.VERTICAL)
        ):
            board = deepcopy(self.board)
            board.shift_vertical_h(
                spaces=self.board.spaces, direction=ShiftDirection.LEFT
            )
            boards.append(board)
        elif east_a_tile == Tile.KHUNPAN and east_b_tile == Tile.KHUNPAN:
            board = deepcopy(self.board)
            board.shift_khunpan(spaces=self.board.spaces, direction=ShiftDirection.LEFT)
            boards.append(board)
        if east_a_tile == Tile.SINGLE:
            board = deepcopy(self.board)
            board.shift_single(spaces[0], ShiftDirection.LEFT)
            board.shift_single(spaces[1], ShiftDirection.DOWN)
            boards.append(board)
        if east_b_tile == Tile.SINGLE:
            board = deepcopy(self.board)
            board.shift_single(spaces[1], ShiftDirection.LEFT)
            board.shift_single(spaces[0], ShiftDirection.UP)
            boards.append(board)
        return boards

    def get_adjacent_moves_horizontal(self, boards: list[Board]) -> list[Board]:
        """Get the boards after adjacent moves in the horizontal direction."""
        spaces = Board.align_spaces(self.board.spaces, ShiftDirection.RIGHT)
        northleft = Board.get_north_coord(spaces[0])
        northright = Board.get_north_coord(spaces[1])
        northwest = Board.get_west_coord(northleft)
        northeast = Board.get_east_coord(northright)
        north_a_tile = self.board.get_tile(northleft)
        north_b_tile = self.board.get_tile(northright)
        nw_tile = self.board.get_tile(northwest)
        ne_tile = self.board.get_tile(northeast)
        if (
            north_a_tile == Tile.HORIZONTAL
            and north_b_tile == Tile.HORIZONTAL
            and not (nw_tile == Tile.HORIZONTAL and ne_tile == Tile.HORIZONTAL)
        ):
            board = deepcopy(self.board)
            board.shift_horizontal_v(
                spaces=self.board.spaces, direction=ShiftDirection.DOWN
            )
            boards.append(board)
        elif north_a_tile == Tile.KHUNPAN and north_b_tile == Tile.KHUNPAN:
            board = deepcopy(self.board)
            board.shift_khunpan(spaces=self.board.spaces, direction=ShiftDirection.DOWN)
            boards.append(board)
        if north_a_tile == Tile.SINGLE:
            board = deepcopy(self.board)
            board.shift_single(spaces[0], ShiftDirection.DOWN)
            board.shift_single(spaces[1], ShiftDirection.RIGHT)
            boards.append(board)
        if north_b_tile == Tile.SINGLE:
            board = deepcopy(self.board)
            board.shift_single(spaces[1], ShiftDirection.DOWN)
            board.shift_single(spaces[0], ShiftDirection.LEFT)
            boards.append(board)
        southleft = Board.get_south_coord(spaces[0])
        southright = Board.get_south_coord(spaces[1])
        southwest = Board.get_west_coord(southleft)
        southeast = Board.get_east_coord(southright)
        south_a_tile = self.board.get_tile(southleft)
        south_b_tile = self.board.get_tile(southright)
        sw_tile = self.board.get_tile(southwest)
        se_tile = self.board.get_tile(southeast)
        if (
            south_a_tile == Tile.HORIZONTAL
            and south_b_tile == Tile.HORIZONTAL
            and not (sw_tile == Tile.HORIZONTAL and se_tile == Tile.HORIZONTAL)
        ):
            board = deepcopy(self.board)
            board.shift_horizontal_v(
                spaces=self.board.spaces, direction=ShiftDirection.UP
            )
            boards.append(board)
        elif south_a_tile == Tile.KHUNPAN and south_b_tile == Tile.KHUNPAN:
            board = deepcopy(self.board)
            board.shift_khunpan(spaces=self.board.spaces, direction=ShiftDirection.UP)
            boards.append(board)
        if south_a_tile == Tile.SINGLE:
            board = deepcopy(self.board)
            board.shift_single(spaces[0], ShiftDirection.UP)
            board.shift_single(spaces[1], ShiftDirection.RIGHT)
            boards.append(board)
        if south_b_tile == Tile.SINGLE:
            board = deepcopy(self.board)
            board.shift_single(spaces[1], ShiftDirection.UP)
            board.shift_single(spaces[0], ShiftDirection.LEFT)
            boards.append(board)
        return boards

    def get_moves(self) -> list[Board]:
        """Get the possible moves for the board."""
        adjacent_spaces = Adjacency.NON
        boards = []
        for space in self.board.spaces:
            east_tile = self.board.get_tile(Board.get_east_coord(space))
            boards = self.get_move(boards, space, east_tile, ShiftDirection.LEFT)

            west_tile = self.board.get_tile(Board.get_west_coord(space))
            boards = self.get_move(boards, space, west_tile, ShiftDirection.RIGHT)

            north_tile = self.board.get_tile(Board.get_north_coord(space))
            boards = self.get_move(boards, space, north_tile, ShiftDirection.DOWN)

            south_tile = self.board.get_tile(Board.get_south_coord(space))
            boards = self.get_move(boards, space, south_tile, ShiftDirection.UP)

            if east_tile == Tile.EMPTY:
                adjacent_spaces = Adjacency.HORIZONTAL
                boards = self.get_double_move(boards, west_tile, ShiftDirection.RIGHT)
            elif west_tile == Tile.EMPTY:
                adjacent_spaces = Adjacency.HORIZONTAL
                boards = self.get_double_move(boards, east_tile, ShiftDirection.LEFT)
            elif north_tile == Tile.EMPTY:
                adjacent_spaces = Adjacency.VERTICAL
                boards = self.get_double_move(boards, south_tile, ShiftDirection.UP)
            elif south_tile == Tile.EMPTY:
                adjacent_spaces = Adjacency.VERTICAL
                boards = self.get_double_move(boards, north_tile, ShiftDirection.DOWN)

        if adjacent_spaces == Adjacency.HORIZONTAL:
            boards = self.get_adjacent_moves_horizontal(boards)
        elif adjacent_spaces == Adjacency.VERTICAL:
            boards = self.get_adjacent_moves_vertical(boards)

        return boards


class KhunPanEscape:
    """Class for the Khun Pan escape game."""

    def __init__(self, board: GameBoard = None) -> None:
        """Initialize the game."""
        self.moves = deque()
        self.processed_moves = []
        self.serialized_moves = []
        start = KhunPanBoard()
        if board is not None:
            start = KhunPanBoard(board=Board(board))
        self.add_move(start)

    def add_move(self, move: KhunPanBoard) -> None:
        """Add a move to the game."""
        ser = move.board.encode()
        if bisect.bisect(self.serialized_moves, ser) == bisect.bisect_left(
            self.serialized_moves, ser
        ):
            bisect.insort(self.serialized_moves, ser)
            self.moves.append(move)

    def solve(self) -> None:
        """Solve the game."""
        won = False
        while not won and len(self.moves) > 0:
            board = self.moves.popleft()
            self.processed_moves.append(board)
            moves = board.get_moves()
            for m in moves:
                if m.check_win_condition():
                    self.processed_moves.append(
                        KhunPanBoard(board=m, predecessor=board.id)
                    )
                    won = True
                    break
                else:
                    self.add_move(KhunPanBoard(board=m, predecessor=board.id))
        print("Game finished.")

    def get_solution(self, win_board: KhunPanBoard = None) -> list[KhunPanBoard]:
        """Get the solution for the game."""
        solution = []
        move = self.processed_moves[-1]
        if win_board is not None:
            move = win_board
        while move.predecessor > 0:
            solution.append(move)
            move = self.processed_moves[
                bisect.bisect_left(
                    self.processed_moves, move.predecessor, key=lambda x: x.id
                )
            ]
        solution.reverse()
        return solution

    @staticmethod
    def print_solution(solution: list[KhunPanBoard]) -> None:
        """Print the solution for the game."""
        for i, move in enumerate(solution):
            print(f"Move {i + 1:d}")
            print(move.board)
            print()


if __name__ == "__main__":
    import time

    start = time.time()
    game = KhunPanEscape()
    game.solve()
    end1 = time.time()
    KhunPanEscape.print_solution(game.get_solution())
    end2 = time.time()
    print(end1 - start)
    print(end2 - start)
