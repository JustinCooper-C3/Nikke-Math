"""
Solution Module - Result of strategy computation and cached solution management.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .board import BoardState
from .move import Move


@dataclass
class SolutionMetrics:
    """
    Performance metrics for solution computation.

    Attributes:
        computation_time_ms: Time taken in milliseconds
        states_explored: Number of board states evaluated
        pruned_branches: Number of branches pruned (for search strategies)
        strategy_name: Name of strategy that computed this solution
    """
    computation_time_ms: float = 0.0
    states_explored: int = 0
    pruned_branches: int = 0
    strategy_name: str = ""


@dataclass
class Solution:
    """
    Result of a strategy computation.

    Attributes:
        moves: Ordered sequence of moves to execute
        total_cleared: Total cells cleared by all moves
        is_complete: True if no more valid moves remain
        was_cancelled: True if stopped before completion
        metrics: Performance statistics
        board_states: Board state after each move (first is initial)
    """
    moves: List[Move] = field(default_factory=list)
    total_cleared: int = 0
    is_complete: bool = False
    was_cancelled: bool = False
    metrics: SolutionMetrics = field(default_factory=SolutionMetrics)
    board_states: List[BoardState] = field(default_factory=list)

    @property
    def move_count(self) -> int:
        """Number of moves in solution."""
        return len(self.moves)

    @property
    def has_moves(self) -> bool:
        """Check if solution has any moves."""
        return len(self.moves) > 0

    def get_move(self, index: int) -> Move:
        """
        Get move at specific index.

        Args:
            index: Move index (0-based)

        Returns:
            Move at index

        Raises:
            IndexError: If index out of range
        """
        return self.moves[index]

    def get_board_after_move(self, index: int) -> BoardState:
        """
        Get board state after executing move at index.

        Args:
            index: Move index (0-based)

        Returns:
            BoardState after move (index+1 in board_states)

        Raises:
            IndexError: If index out of range
        """
        return self.board_states[index + 1]


@dataclass
class CachedSolution:
    """
    Full solution with move queue and expected board states for cache-based playback.

    Wraps a Solution object and tracks progress through the move sequence,
    enabling move-by-move validation and cache invalidation.

    Attributes:
        solution: The complete solution from optimizer
        move_index: Current position in move sequence (0 = first move)
        created_at: Timestamp for cache staleness detection
    """
    solution: Solution
    move_index: int = 0
    created_at: float = field(default_factory=time.perf_counter)

    @property
    def current_move(self) -> Optional[Move]:
        """Get next move to display, or None if exhausted."""
        if self.move_index < len(self.solution.moves):
            return self.solution.moves[self.move_index]
        return None

    @property
    def expected_cells_to_clear(self) -> Set[Tuple[int, int]]:
        """Get cells expected to be cleared by current move."""
        move = self.current_move
        if move is None:
            return set()
        return set(move.cells)

    @property
    def expected_board_before(self) -> Optional[BoardState]:
        """Get expected board state before current move executes."""
        if self.move_index < len(self.solution.board_states):
            return self.solution.board_states[self.move_index]
        return None

    @property
    def expected_board_after(self) -> Optional[BoardState]:
        """Get expected board state after current move executes."""
        if self.move_index + 1 < len(self.solution.board_states):
            return self.solution.board_states[self.move_index + 1]
        return None

    @property
    def is_exhausted(self) -> bool:
        """True if all moves have been consumed."""
        return self.move_index >= len(self.solution.moves)

    @property
    def moves_remaining(self) -> int:
        """Number of moves left in the solution."""
        return max(0, len(self.solution.moves) - self.move_index)

    @property
    def total_moves(self) -> int:
        """Total number of moves in the solution."""
        return len(self.solution.moves)

    @property
    def age_seconds(self) -> float:
        """Time since cache was created."""
        return time.perf_counter() - self.created_at

    def advance(self) -> Optional[Move]:
        """
        Move to next move in sequence.

        Returns:
            The move that was just completed, or None if exhausted
        """
        if self.is_exhausted:
            return None
        completed_move = self.current_move
        self.move_index += 1
        return completed_move

    def peek_moves(self, count: int = 3) -> List[Move]:
        """
        Preview upcoming moves without advancing.

        Args:
            count: Number of moves to preview

        Returns:
            List of upcoming moves (may be shorter than count)
        """
        start = self.move_index
        end = min(start + count, len(self.solution.moves))
        return self.solution.moves[start:end]

    def validate_board_match(
        self,
        actual: BoardState,
        unstable_cells: Optional[Set[Tuple[int, int]]] = None
    ) -> bool:
        """
        Check if actual board matches expected board after current move.

        Ignores cells that are currently unstable (flickering due to selection).

        Args:
            actual: Actual board state from OCR
            unstable_cells: Cells to ignore in comparison (selection flutter)

        Returns:
            True if boards match (excluding unstable cells)
        """
        expected = self.expected_board_after
        if expected is None:
            return False

        # Get differing cells (diff returns List, convert to set)
        diff = set(actual.diff(expected))

        # Remove unstable cells from consideration
        if unstable_cells:
            diff = diff - unstable_cells

        return len(diff) == 0

    def validate_cells_cleared(self, actual: BoardState) -> bool:
        """
        Fast-path validation: check if expected cells are now empty.

        Args:
            actual: Actual board state from OCR

        Returns:
            True if all expected cells are now None/empty
        """
        expected_cleared = self.expected_cells_to_clear
        if not expected_cleared:
            return False

        for r, c in expected_cleared:
            cell_value = actual.get_cell(r, c)
            if cell_value is not None and cell_value != 0:
                return False
        return True
