"""
Solution Manager Module - UI state machine for cached solution flow.

This module provides the SolutionManager which handles the UI-level
state machine for displaying moves from a cached solution and validating clears.

Uses tiered validation:
  - Tier 1 (Fast): Check if expected cells cleared
  - Tier 2 (Stability): Wait for stable frames to filter flutter
  - Tier 3 (Full): Compare board against expected state

For the core solving logic, see the src.solver package.
"""

from enum import Enum, auto
from typing import List, Tuple, Optional, Set
import logging

from src.solver import (
    BoardState, Move, Solution, CachedSolution,
    SolverStrategy, SolutionContext, create_strategy
)

logger = logging.getLogger(__name__)


__all__ = [
    "SolutionState",
    "SolutionManager",
]


class SolutionState(Enum):
    """
    State machine states for cached solution flow.

    States:
        WAITING_STABLE: Collecting frames to establish stable board
        COMPUTING_SOLUTION: Computing full solution (all moves)
        MOVE_DISPLAYED: Move is shown, waiting for user to execute
        VALIDATING_CLEAR: Checking if expected cells were cleared
    """
    WAITING_STABLE = auto()
    COMPUTING_SOLUTION = auto()
    MOVE_DISPLAYED = auto()
    VALIDATING_CLEAR = auto()


class SolutionManager:
    """
    Cached solution manager with tiered validation.

    Implements a state machine that:
    1. Waits for stable board (3 consecutive identical frames)
    2. Computes FULL solution using the selected strategy
    3. Displays moves one at a time from cache
    4. Validates clears using tiered approach (fast/stability/full)
    5. Only recomputes when board diverges from expected state

    State Flow:
        WAITING_STABLE -> COMPUTING_SOLUTION -> MOVE_DISPLAYED -> VALIDATING_CLEAR
                ^                                      |                |
                |                                      +-- (pop next) --+
                |______________________________________________________|
                    (cache invalid OR exhausted OR timeout)
    """

    # Cache timeout in seconds (recompute if cache older than this)
    CACHE_TIMEOUT_SEC = 30.0

    def __init__(self, stability_threshold: int = 3, timeout_frames: int = 30,
                 strategy_name: str = "greedy"):
        """
        Initialize solution manager.

        Args:
            stability_threshold: Consecutive identical frames required (default 3)
            timeout_frames: Frames to wait for clear before recalc (default 30, ~3s at 10 FPS)
            strategy_name: Name of solving strategy to use (default "greedy")
        """
        self.stability_threshold = stability_threshold
        self.timeout_frames = timeout_frames

        # Strategy
        self._strategy: SolverStrategy = create_strategy(strategy_name)

        # State machine
        self._state = SolutionState.WAITING_STABLE

        # Frame buffer for stability detection
        self._frame_buffer: List[BoardState] = []

        # Board tracking
        self._current_board: Optional[BoardState] = None
        self._last_stable_board: Optional[BoardState] = None

        # Cached solution (full solution with move queue)
        self._cached_solution: Optional[CachedSolution] = None

        # Current move tracking (for backward compatibility)
        self._current_move: Optional[Move] = None
        self._expected_cleared: Set[Tuple[int, int]] = set()
        self._frames_since_display: int = 0

        # Unstable cell tracking (for OCR noise detection)
        self._unstable_cells: Set[Tuple[int, int]] = set()

    @property
    def strategy(self) -> SolverStrategy:
        """Get current solving strategy."""
        return self._strategy

    @property
    def strategy_name(self) -> str:
        """Get current strategy name."""
        return self._strategy.name

    def set_strategy(self, strategy_name: str) -> None:
        """
        Change the solving strategy.

        Args:
            strategy_name: Name of strategy to use
        """
        self._strategy = create_strategy(strategy_name)
        logger.info(f"Strategy changed to: {strategy_name}")

    @property
    def state(self) -> SolutionState:
        """Get current state machine state."""
        return self._state

    @property
    def current_move(self) -> Optional[Move]:
        """Get the current move being displayed."""
        return self._current_move

    @property
    def expected_cleared(self) -> Set[Tuple[int, int]]:
        """Get cells expected to be cleared by current move."""
        return self._expected_cleared.copy()

    @property
    def is_stable(self) -> bool:
        """Check if board is currently stable."""
        return self._state in (SolutionState.MOVE_DISPLAYED, SolutionState.VALIDATING_CLEAR)

    @property
    def current_board(self) -> Optional[BoardState]:
        """Get current board state."""
        return self._current_board

    @property
    def unstable_cells(self) -> Set[Tuple[int, int]]:
        """Get cells that disagreed across recent frames."""
        return self._unstable_cells.copy()

    @property
    def cached_solution(self) -> Optional[CachedSolution]:
        """Get current cached solution."""
        return self._cached_solution

    @property
    def moves_remaining(self) -> int:
        """Number of moves remaining in cached solution."""
        if self._cached_solution is None:
            return 0
        return self._cached_solution.moves_remaining

    @property
    def total_moves(self) -> int:
        """Total moves in current solution."""
        if self._cached_solution is None:
            return 0
        return self._cached_solution.total_moves

    def peek_next_moves(self, count: int = 3) -> List[Move]:
        """
        Preview upcoming moves without advancing.

        Args:
            count: Number of moves to preview

        Returns:
            List of upcoming moves
        """
        if self._cached_solution is None:
            return []
        return self._cached_solution.peek_moves(count)

    def invalidate_cache(self) -> None:
        """Force cache invalidation and trigger recomputation."""
        logger.info("Cache invalidated manually")
        self._cached_solution = None
        self._transition_to_waiting_stable()

    def update(self, board: BoardState) -> bool:
        """
        Update with new board state and process state machine.

        Args:
            board: New board state from OCR

        Returns:
            True if a move is ready to display, False otherwise
        """
        self._current_board = board

        if self._state == SolutionState.WAITING_STABLE:
            return self._handle_waiting_stable(board)
        elif self._state == SolutionState.COMPUTING_SOLUTION:
            return self._handle_computing_solution(board)
        elif self._state == SolutionState.MOVE_DISPLAYED:
            return self._handle_move_displayed(board)
        elif self._state == SolutionState.VALIDATING_CLEAR:
            return self._handle_validating_clear(board)

        return False

    def _handle_waiting_stable(self, board: BoardState) -> bool:
        """Handle WAITING_STABLE state - collect frames for stability."""
        self._frame_buffer.append(board)

        if len(self._frame_buffer) > self.stability_threshold:
            self._frame_buffer.pop(0)

        if len(self._frame_buffer) < self.stability_threshold:
            logger.debug(f"State[WAITING_STABLE]: collecting frames ({len(self._frame_buffer)}/{self.stability_threshold})")
            return False

        first_board = self._frame_buffer[0]
        all_same = all(b == first_board for b in self._frame_buffer)

        if not all_same:
            self._unstable_cells = self._find_unstable_cells()
            logger.debug(f"State[WAITING_STABLE]: frames differ, {len(self._unstable_cells)} unstable cells")
            return self._current_move is not None

        self._unstable_cells = set()
        self._last_stable_board = first_board

        logger.info("State[WAITING_STABLE]: board stable, transitioning to COMPUTING_SOLUTION")
        self._state = SolutionState.COMPUTING_SOLUTION
        return self._handle_computing_solution(board)

    def _handle_computing_solution(self, board: BoardState) -> bool:
        """Handle COMPUTING_SOLUTION state - compute full solution and cache it."""
        import time

        if self._last_stable_board is None:
            logger.warning("State[COMPUTING_SOLUTION]: no stable board, returning to WAITING_STABLE")
            self._state = SolutionState.WAITING_STABLE
            return False

        start_time = time.perf_counter()

        # Create context and compute full solution
        context = SolutionContext(
            board=self._last_stable_board,
            timeout_sec=self._strategy.timeout_sec
        )
        solution = self._strategy.solve(context)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if not solution.has_moves:
            logger.info(f"State[COMPUTING_SOLUTION]: no valid moves found ({elapsed_ms:.1f}ms)")
            self._cached_solution = None
            self._current_move = None
            self._expected_cleared = set()
            self._state = SolutionState.WAITING_STABLE
            self._frame_buffer.clear()
            return False

        # Cache the full solution
        self._cached_solution = CachedSolution(solution=solution)

        logger.info(
            f"State[COMPUTING_SOLUTION]: computed {solution.move_count} moves, "
            f"{solution.total_cleared} total cells ({elapsed_ms:.1f}ms)"
        )

        # Set up first move for display
        return self._display_current_move()

    def _handle_move_displayed(self, board: BoardState) -> bool:
        """Handle MOVE_DISPLAYED state - wait for board change."""
        # Check for cache timeout (stale solution)
        if self._cached_solution is not None:
            if self._cached_solution.age_seconds > self.CACHE_TIMEOUT_SEC:
                logger.warning(f"State[MOVE_DISPLAYED]: cache expired ({self._cached_solution.age_seconds:.1f}s), recomputing")
                return self._invalidate_and_recompute(board)

        if self._last_stable_board is not None and board != self._last_stable_board:
            logger.info("State[MOVE_DISPLAYED]: board changed, transitioning to VALIDATING_CLEAR")
            self._state = SolutionState.VALIDATING_CLEAR
            self._frame_buffer.clear()
            return self._handle_validating_clear(board)

        self._frames_since_display += 1

        if self._frames_since_display >= self.timeout_frames:
            logger.warning(f"State[MOVE_DISPLAYED]: timeout after {self._frames_since_display} frames, recalculating")
            self._cached_solution = None  # Clear stale cache
            self._transition_to_waiting_stable()
            return False

        return True

    def _handle_validating_clear(self, board: BoardState) -> bool:
        """
        Handle VALIDATING_CLEAR state - tiered validation of cell clearing.

        Tier 1 (Fast): Check if expected cells cleared immediately
        Tier 2 (Stability): Wait for stable frames to filter flutter
        Tier 3 (Full): Compare stable board against expected state
        """
        # Tier 1: Fast-path check - did expected cells clear?
        if self._cached_solution is not None:
            if self._cached_solution.validate_cells_cleared(board):
                logger.info("State[VALIDATING_CLEAR]: Tier 1 PASS - expected cells cleared")
                return self._advance_to_next_move(board)

        # Tier 2: Wait for stability (filter flutter)
        self._frame_buffer.append(board)

        if len(self._frame_buffer) > self.stability_threshold:
            self._frame_buffer.pop(0)

        if len(self._frame_buffer) < self.stability_threshold:
            logger.debug(f"State[VALIDATING_CLEAR]: Tier 2 - waiting for stability ({len(self._frame_buffer)}/{self.stability_threshold})")
            return self._current_move is not None

        first_board = self._frame_buffer[0]
        all_same = all(b == first_board for b in self._frame_buffer)

        if not all_same:
            self._unstable_cells = self._find_unstable_cells()
            logger.debug(f"State[VALIDATING_CLEAR]: Tier 2 - still stabilizing, {len(self._unstable_cells)} unstable cells")
            return self._current_move is not None

        self._unstable_cells = set()

        # Tier 3: Full board comparison (with unstable cell filtering)
        if self._cached_solution is not None:
            if self._cached_solution.validate_board_match(first_board, self._unstable_cells):
                logger.info("State[VALIDATING_CLEAR]: Tier 3 PASS - board matches expected")
                return self._advance_to_next_move(first_board)

        # Tier 3 failed - check what happened
        cells_cleared = set()
        for r, c in self._expected_cleared:
            cell_value = first_board.get_cell(r, c)
            if cell_value is None or cell_value == 0:
                cells_cleared.add((r, c))

        if cells_cleared == self._expected_cleared:
            # Cells cleared but board doesn't match expected (user did extra moves?)
            logger.info(f"State[VALIDATING_CLEAR]: cells cleared but board differs from expected, invalidating cache")
            return self._invalidate_and_recompute(first_board)

        elif len(cells_cleared) > 0:
            # Partial clear or different cells cleared
            logger.info(f"State[VALIDATING_CLEAR]: {len(cells_cleared)}/{len(self._expected_cleared)} cells cleared (unexpected), invalidating cache")
            return self._invalidate_and_recompute(first_board)

        else:
            # No expected cells cleared
            if first_board != self._last_stable_board:
                # Board changed but not as expected - invalidate
                logger.info("State[VALIDATING_CLEAR]: board changed unexpectedly, invalidating cache")
                return self._invalidate_and_recompute(first_board)
            else:
                # Board unchanged - go back to displaying move
                logger.debug("State[VALIDATING_CLEAR]: no change detected, returning to MOVE_DISPLAYED")
                self._state = SolutionState.MOVE_DISPLAYED
                return True

    def _transition_to_waiting_stable(self) -> None:
        """Helper to transition back to WAITING_STABLE state."""
        self._state = SolutionState.WAITING_STABLE
        self._frame_buffer.clear()
        self._current_move = None
        self._expected_cleared = set()
        self._frames_since_display = 0

    def _display_current_move(self) -> bool:
        """
        Set up current move from cache for display.

        Returns:
            True if a move is ready to display, False if cache exhausted
        """
        if self._cached_solution is None or self._cached_solution.is_exhausted:
            logger.info("Cache exhausted, no more moves")
            self._current_move = None
            self._expected_cleared = set()
            self._transition_to_waiting_stable()
            return False

        move = self._cached_solution.current_move
        self._current_move = move
        self._expected_cleared = self._cached_solution.expected_cells_to_clear
        self._frames_since_display = 0

        self._state = SolutionState.MOVE_DISPLAYED

        remaining = self._cached_solution.moves_remaining
        total = self._cached_solution.total_moves
        logger.info(
            f"Displaying move {total - remaining + 1}/{total}: "
            f"clearing {move.cell_count} cells"
        )

        return True

    def _advance_to_next_move(self, new_board: BoardState) -> bool:
        """
        Advance cache to next move after successful validation.

        Args:
            new_board: The validated stable board state

        Returns:
            True if next move is ready, False if solution complete
        """
        self._last_stable_board = new_board
        self._frame_buffer.clear()

        if self._cached_solution is not None:
            self._cached_solution.advance()

        return self._display_current_move()

    def _invalidate_and_recompute(self, new_board: BoardState) -> bool:
        """
        Invalidate cache and recompute solution from current board.

        Args:
            new_board: The current board state to solve from

        Returns:
            True if new move ready, False otherwise
        """
        logger.info("Cache invalidated, recomputing solution")
        self._cached_solution = None
        self._last_stable_board = new_board
        self._frame_buffer.clear()
        self._state = SolutionState.COMPUTING_SOLUTION
        return self._handle_computing_solution(new_board)

    def _find_unstable_cells(self) -> Set[Tuple[int, int]]:
        """Find cells that have different values across frames in buffer."""
        if len(self._frame_buffer) < 2:
            return set()

        unstable: Set[Tuple[int, int]] = set()
        first_board = self._frame_buffer[0]
        rows = first_board.rows
        cols = first_board.cols

        for other_board in self._frame_buffer[1:]:
            if other_board.rows != rows or other_board.cols != cols:
                return {(r, c) for r in range(rows) for c in range(cols)}

        for r in range(rows):
            for c in range(cols):
                first_val = first_board.get_cell(r, c)
                for other_board in self._frame_buffer[1:]:
                    if other_board.get_cell(r, c) != first_val:
                        unstable.add((r, c))
                        break

        return unstable

    def move_uses_unstable_cells(self, move: Move) -> bool:
        """Check if a move uses any cells that were unstable across frames."""
        for cell in move.cells:
            if cell in self._unstable_cells:
                return True
        return False

    def reset(self) -> None:
        """Reset solution manager to initial state."""
        self._state = SolutionState.WAITING_STABLE
        self._frame_buffer = []
        self._current_board = None
        self._last_stable_board = None
        self._cached_solution = None
        self._current_move = None
        self._expected_cleared = set()
        self._frames_since_display = 0
        self._unstable_cells = set()
        logger.info("SolutionManager reset")

    def get_state_string(self) -> str:
        """Get human-readable state string for UI display."""
        state_strings = {
            SolutionState.WAITING_STABLE: "Stabilizing",
            SolutionState.COMPUTING_SOLUTION: "Computing",
            SolutionState.MOVE_DISPLAYED: "Move Ready",
            SolutionState.VALIDATING_CLEAR: "Validating"
        }
        base = state_strings.get(self._state, "Unknown")

        # Add move progress if available
        if self._cached_solution is not None and not self._cached_solution.is_exhausted:
            remaining = self._cached_solution.moves_remaining
            total = self._cached_solution.total_moves
            current = total - remaining + 1
            return f"{base} ({current}/{total})"

        return base
