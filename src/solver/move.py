"""
Move Module - Represents a valid rectangle move on the board.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Move:
    """
    Represents a valid rectangle move on the board.

    A move defines a rectangular region where all contained cells
    sum to exactly 10. The rectangle is defined by top-left and
    bottom-right corners.

    Attributes:
        r1: Top row index (inclusive)
        c1: Left column index (inclusive)
        r2: Bottom row index (inclusive)
        c2: Right column index (inclusive)
        cells: Tuple of (row, col) tuples for cells with values
        total: Sum of all cell values (should be 10)
    """
    r1: int
    c1: int
    r2: int
    c2: int
    cells: Tuple[Tuple[int, int], ...]
    total: int

    @classmethod
    def create(cls, r1: int, c1: int, r2: int, c2: int,
               cells: List[Tuple[int, int]], total: int) -> 'Move':
        """
        Create a Move with list cells converted to tuple.

        Args:
            r1: Top row
            c1: Left column
            r2: Bottom row
            c2: Right column
            cells: List of (row, col) cell positions
            total: Sum of cells (should be 10)

        Returns:
            Move instance
        """
        return cls(r1=r1, c1=c1, r2=r2, c2=c2,
                   cells=tuple(cells), total=total)

    @property
    def cell_count(self) -> int:
        """Number of cells cleared by this move."""
        return len(self.cells)

    @property
    def area(self) -> int:
        """Area of the rectangle (including empty cells)."""
        return (self.r2 - self.r1 + 1) * (self.c2 - self.c1 + 1)

    @property
    def tight_bounds(self) -> Tuple[int, int, int, int]:
        """
        Calculate tight bounding box from actual non-zero cells.

        Returns:
            (min_row, min_col, max_row, max_col) - bounds containing only cells with values
        """
        if not self.cells:
            # Fallback to original bounds if no cells
            return (self.r1, self.c1, self.r2, self.c2)

        rows = [cell[0] for cell in self.cells]
        cols = [cell[1] for cell in self.cells]

        return (min(rows), min(cols), max(rows), max(cols))

    def __hash__(self):
        """Enable using Move in sets and as dict keys."""
        return hash((self.r1, self.c1, self.r2, self.c2, self.cells))
