"""
Solution Manager Module - UI state machine for single-move solution flow.

This module provides the SolutionManager which handles the UI-level
state machine for displaying one move at a time and validating clears.

For the core solving logic, see the src.solver package.
"""

from enum import Enum, auto
from typing import List, Tuple, Optional, Set
import logging

from src.solver import BoardState, Move, SolverStrategy, create_strategy

logger = logging.getLogger(__name__)


__all__ = [
    "SolutionState",
    "SolutionManager",
]


class SolutionState(Enum):
    """
    State machine states for single-move solution flow.

    States:
        WAITING_STABLE: Collecting frames to establish stable board
        COMPUTING_MOVE: Finding the best next move
        MOVE_DISPLAYED: Move is shown, waiting for user to execute
        VALIDATING_CLEAR: Checking if expected cells were cleared
    """
    WAITING_STABLE = auto()
    COMPUTING_MOVE = auto()
    MOVE_DISPLAYED = auto()
    VALIDATING_CLEAR = auto()


class SolutionManager:
    """
    Single-move-at-a-time solution manager with clear validation.

    Implements a state machine that:
    1. Waits for stable board (3 consecutive identical frames)
    2. Computes ONE best move using the selected strategy
    3. Displays move and tracks expected cells to clear
    4. Validates that cells were cleared before computing next move
    5. Handles timeouts and unexpected board changes

    State Flow:
        WAITING_STABLE -> COMPUTING_MOVE -> MOVE_DISPLAYED -> VALIDATING_CLEAR
                ^                                                    |
                |____________________________________________________|
                    (cells cleared OR timeout OR unexpected change)
    """

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

        # Current move tracking
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
        elif self._state == SolutionState.COMPUTING_MOVE:
            return self._handle_computing_move(board)
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

        logger.info("State[WAITING_STABLE]: board stable, transitioning to COMPUTING_MOVE")
        self._state = SolutionState.COMPUTING_MOVE
        return self._handle_computing_move(board)

    def _handle_computing_move(self, board: BoardState) -> bool:
        """Handle COMPUTING_MOVE state - find best next move."""
        import time

        if self._last_stable_board is None:
            logger.warning("State[COMPUTING_MOVE]: no stable board, returning to WAITING_STABLE")
            self._state = SolutionState.WAITING_STABLE
            return False

        start_time = time.perf_counter()

        # Use strategy to find best move
        best_move = self._strategy.find_best_move(self._last_stable_board)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if best_move is None:
            logger.info(f"State[COMPUTING_MOVE]: no valid moves found ({elapsed_ms:.1f}ms)")
            self._current_move = None
            self._expected_cleared = set()
            self._state = SolutionState.WAITING_STABLE
            self._frame_buffer.clear()
            return False

        logger.info(f"State[COMPUTING_MOVE]: found move clearing {best_move.cell_count} cells ({elapsed_ms:.1f}ms)")

        self._current_move = best_move
        self._expected_cleared = set(best_move.cells)
        self._frames_since_display = 0

        self._state = SolutionState.MOVE_DISPLAYED
        logger.info(f"State[COMPUTING_MOVE]: transitioning to MOVE_DISPLAYED, expecting {len(self._expected_cleared)} cells to clear")

        return True

    def _handle_move_displayed(self, board: BoardState) -> bool:
        """Handle MOVE_DISPLAYED state - wait for board change."""
        if self._last_stable_board is not None and board != self._last_stable_board:
            logger.info("State[MOVE_DISPLAYED]: board changed, transitioning to VALIDATING_CLEAR")
            self._state = SolutionState.VALIDATING_CLEAR
            self._frame_buffer.clear()
            return self._handle_validating_clear(board)

        self._frames_since_display += 1

        if self._frames_since_display >= self.timeout_frames:
            logger.warning(f"State[MOVE_DISPLAYED]: timeout after {self._frames_since_display} frames, recalculating")
            self._transition_to_waiting_stable()
            return False

        return True

    def _handle_validating_clear(self, board: BoardState) -> bool:
        """Handle VALIDATING_CLEAR state - check if expected cells cleared."""
        self._frame_buffer.append(board)

        if len(self._frame_buffer) > self.stability_threshold:
            self._frame_buffer.pop(0)

        if len(self._frame_buffer) < self.stability_threshold:
            logger.debug(f"State[VALIDATING_CLEAR]: waiting for stability ({len(self._frame_buffer)}/{self.stability_threshold})")
            return self._current_move is not None

        first_board = self._frame_buffer[0]
        all_same = all(b == first_board for b in self._frame_buffer)

        if not all_same:
            self._unstable_cells = self._find_unstable_cells()
            logger.debug(f"State[VALIDATING_CLEAR]: still stabilizing, {len(self._unstable_cells)} unstable cells")
            return self._current_move is not None

        self._unstable_cells = set()

        cells_cleared = set()
        for r, c in self._expected_cleared:
            if first_board.get_cell(r, c) is None:
                cells_cleared.add((r, c))

        if cells_cleared == self._expected_cleared:
            logger.info(f"State[VALIDATING_CLEAR]: all {len(cells_cleared)} expected cells cleared, computing next move")
            self._last_stable_board = first_board
            self._current_move = None
            self._expected_cleared = set()
            self._state = SolutionState.COMPUTING_MOVE
            return self._handle_computing_move(first_board)

        elif len(cells_cleared) > 0:
            logger.info(f"State[VALIDATING_CLEAR]: {len(cells_cleared)}/{len(self._expected_cleared)} cells cleared (partial/different), accepting new state")
            self._last_stable_board = first_board
            self._current_move = None
            self._expected_cleared = set()
            self._state = SolutionState.COMPUTING_MOVE
            return self._handle_computing_move(first_board)

        else:
            if first_board != self._last_stable_board:
                logger.info("State[VALIDATING_CLEAR]: board changed but no expected cells cleared, accepting new state")
                self._last_stable_board = first_board
                self._current_move = None
                self._expected_cleared = set()
                self._state = SolutionState.COMPUTING_MOVE
                return self._handle_computing_move(first_board)
            else:
                logger.debug("State[VALIDATING_CLEAR]: no change detected, continuing to wait")
                self._state = SolutionState.MOVE_DISPLAYED
                return True

    def _transition_to_waiting_stable(self) -> None:
        """Helper to transition back to WAITING_STABLE state."""
        self._state = SolutionState.WAITING_STABLE
        self._frame_buffer.clear()
        self._current_move = None
        self._expected_cleared = set()
        self._frames_since_display = 0

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
        self._current_move = None
        self._expected_cleared = set()
        self._frames_since_display = 0
        self._unstable_cells = set()
        logger.info("SolutionManager reset")

    def get_state_string(self) -> str:
        """Get human-readable state string for UI display."""
        state_strings = {
            SolutionState.WAITING_STABLE: "Stabilizing",
            SolutionState.COMPUTING_MOVE: "Computing",
            SolutionState.MOVE_DISPLAYED: "Move Ready",
            SolutionState.VALIDATING_CLEAR: "Validating"
        }
        return state_strings.get(self._state, "Unknown")
