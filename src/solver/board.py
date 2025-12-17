"""
Board State Module - Immutable board representation for Sum to 10 puzzle.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .move import Move


@dataclass(frozen=True)
class BoardState:
    """
    Immutable board state representation.

    Uses tuple-of-tuples for hashability and immutability.
    Board cells contain integers 1-9 or None for empty cells.

    Attributes:
        grid: Tuple of tuples representing the board state
              Each cell is either an int (1-9) or None (empty)
    """
    grid: Tuple[Tuple[Optional[int], ...], ...]

    @classmethod
    def from_ocr(cls, ocr_result: List[List[Optional[int]]]) -> 'BoardState':
        """
        Create BoardState from OCR result (2D list).

        Args:
            ocr_result: 2D list of integers or None values

        Returns:
            BoardState instance with immutable grid
        """
        grid = tuple(tuple(row) for row in ocr_result)
        return cls(grid=grid)

    @classmethod
    def from_grid(cls, grid: Tuple[Tuple[Optional[int], ...], ...]) -> 'BoardState':
        """
        Create BoardState from existing grid tuple.

        Args:
            grid: Tuple of tuples representing board state

        Returns:
            BoardState instance
        """
        return cls(grid=grid)

    @classmethod
    def from_2d_list(cls, grid: List[List[Optional[int]]]) -> 'BoardState':
        """
        Create BoardState from 2D list (alias for from_ocr).

        Args:
            grid: 2D list of integers or None values

        Returns:
            BoardState instance
        """
        return cls.from_ocr(grid)

    def diff(self, other: 'BoardState') -> List[Tuple[int, int]]:
        """
        Find cells that differ between this board and another.

        Args:
            other: Another BoardState to compare against

        Returns:
            List of (row, col) tuples where cells differ
        """
        if not isinstance(other, BoardState):
            raise TypeError("Can only diff against another BoardState")

        rows = len(self.grid)
        cols = len(self.grid[0]) if rows > 0 else 0

        differences = []
        for r in range(rows):
            for c in range(cols):
                if self.grid[r][c] != other.grid[r][c]:
                    differences.append((r, c))

        return differences

    def apply_move(self, move: 'Move') -> 'BoardState':
        """
        Apply a move to create a new board state.

        Clears all cells specified in the move and returns a new
        BoardState instance. Original board is unchanged.

        Args:
            move: Move to apply

        Returns:
            New BoardState with cells cleared
        """
        # Convert to mutable list for modification
        new_grid = [list(row) for row in self.grid]

        # Clear all cells in the move
        for r, c in move.cells:
            new_grid[r][c] = None

        # Convert back to immutable tuple
        immutable_grid = tuple(tuple(row) for row in new_grid)
        return BoardState(grid=immutable_grid)

    def count_cells(self) -> int:
        """
        Count non-empty cells on the board.

        Returns:
            Number of cells with values (not None and not 0)
            Note: OCR returns 0 for empty cells, so both None and 0 are excluded
        """
        count = 0
        for row in self.grid:
            for cell in row:
                if cell is not None and cell != 0:
                    count += 1
        return count

    def get_cell(self, row: int, col: int) -> Optional[int]:
        """
        Get value at specific cell position.

        Args:
            row: Row index
            col: Column index

        Returns:
            Cell value (1-9) or None if empty
        """
        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[row]):
            return self.grid[row][col]
        return None

    @property
    def rows(self) -> int:
        """Get number of rows in board."""
        return len(self.grid)

    @property
    def cols(self) -> int:
        """Get number of columns in board."""
        return len(self.grid[0]) if self.rows > 0 else 0

    def __hash__(self):
        """Enable using BoardState as dict key or in sets."""
        return hash(self.grid)

    def __eq__(self, other):
        """Enable board equality comparison."""
        if not isinstance(other, BoardState):
            return False
        return self.grid == other.grid

    def to_list(self) -> List[List[Optional[int]]]:
        """
        Convert to mutable 2D list representation.

        Returns:
            2D list representation of the board
        """
        return [list(row) for row in self.grid]
