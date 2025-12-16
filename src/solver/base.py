"""
Base Strategy Module - Abstract base class for solving strategies.
"""

from abc import ABC, abstractmethod
from typing import List

from .board import BoardState
from .move import Move
from .context import SolutionContext
from .solution import Solution


class SolverStrategy(ABC):
    """
    Abstract base class for all solving strategies.

    Subclasses must implement the solve() method and define
    name and description class attributes.

    Attributes:
        name: Short identifier for the strategy
        description: Human-readable description for UI
        timeout_sec: Default timeout for this strategy
    """
    name: str = "base"
    description: str = "Base strategy"
    timeout_sec: float = 20.0

    @abstractmethod
    def solve(self, context: SolutionContext) -> Solution:
        """
        Compute solution for the given board state.

        Must periodically check context.is_cancelled() and return
        partial solution if True.

        Args:
            context: Solution context with board, cancellation, progress

        Returns:
            Solution with moves and metrics
        """
        pass

    def find_all_valid_moves(self, board: BoardState) -> List[Move]:
        """
        Find all rectangle moves that sum to exactly 10.

        Searches all possible rectangles on the board and identifies
        those where the sum of contained cells equals 10.

        Args:
            board: Current board state

        Returns:
            List of valid Move objects
        """
        moves = []
        rows = board.rows
        cols = board.cols

        # Try all possible rectangle positions
        for r1 in range(rows):
            for c1 in range(cols):
                for r2 in range(r1, rows):
                    for c2 in range(c1, cols):
                        # Calculate sum and collect cells
                        total = 0
                        cells = []

                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                val = board.get_cell(r, c)
                                # Skip empty cells (None) and zero-value cells (OCR returns 0 for empty)
                                if val is not None and val != 0:
                                    total += val
                                    cells.append((r, c))

                        # Valid move if sum is 10 and has at least one cell
                        if total == 10 and len(cells) > 0:
                            move = Move.create(r1, c1, r2, c2, cells, total)
                            moves.append(move)

        return moves

    def find_best_move(self, board: BoardState) -> Move | None:
        """
        Find the single best move on the board (most cells cleared).

        Args:
            board: Current board state

        Returns:
            Best Move object, or None if no valid moves
        """
        moves = self.find_all_valid_moves(board)

        if not moves:
            return None

        # Sort by cell count (descending) and return best
        moves.sort(key=lambda m: m.cell_count, reverse=True)
        return moves[0]

    def _check_cancelled(self, context: SolutionContext) -> bool:
        """
        Convenience method to check cancellation.

        Args:
            context: Solution context

        Returns:
            True if strategy should stop
        """
        return context.is_cancelled()
