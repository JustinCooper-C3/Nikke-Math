"""
Solution Module - Result of strategy computation.
"""

from dataclasses import dataclass, field
from typing import List

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
