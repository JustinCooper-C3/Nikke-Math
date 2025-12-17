"""
Solver Worker Module for Nikke Math Solver

Provides a background QThread worker that manages the capture/solve/display loop.
Communicates with the UI via Qt signals for thread-safe status updates.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

from src.window_capture import WindowCapture
from src.ocr import create_engine, GridInfo, OCRResult
from src.ocr.debug import save_debug_image, DEBUG_DIR
from src.solver import BoardState
from src.solution_manager import SolutionManager, SolutionState


# Configure module logger
logger = logging.getLogger(__name__)


class SolverWorker(QThread):
    """
    Background worker thread for the solver pipeline.

    Runs a continuous loop that:
    1. Monitors for the game window
    2. Captures frames when window is found
    3. (Future) OCR processing
    4. (Future) Solve board
    5. (Future) Update overlay

    Signals:
        status_changed(str): Emitted when worker status changes
        window_changed(str): Emitted when window detection status changes
        board_changed(str): Emitted when board detection status changes
        moves_changed(str): Emitted when move count changes
        error_occurred(str): Emitted when an error occurs

    Example:
        worker = SolverWorker()
        worker.status_changed.connect(ui.set_status)
        worker.window_changed.connect(ui.set_window_info)
        worker.start()
        # ...
        worker.request_stop()
        worker.wait()
    """

    # Signals for UI updates (thread-safe)
    status_changed = pyqtSignal(str)
    window_changed = pyqtSignal(str)
    board_changed = pyqtSignal(str)
    moves_changed = pyqtSignal(int, str, int)  # (total_moves, status, total_cells)
    error_occurred = pyqtSignal(str)
    next_move_ready = pyqtSignal(object)  # Emits Move or None

    # Phase 4 signals
    ocr_status = pyqtSignal(float, int, int)  # confidence%, total_cells, uncertain_count
    fps_update = pyqtSignal(float, float)      # current_fps, fps_cap

    # Phase 6 signals - Overlay support
    # (Move, GridInfo, window_rect, frame, solution_stack, solution_state) or (None, None, None, None, None, None)
    overlay_update = pyqtSignal(object, object, object, object, object, object)

    # Performance constants
    MIN_FRAME_TIME_MS = 100  # 10 FPS cap
    FPS_CAP = 10.0
    FPS_WINDOW_SIZE = 10     # Rolling average window

    # Legacy constant (kept for compatibility)
    POLL_INTERVAL_MS = 100

    def __init__(self, process_name: str = "nikke.exe", ocr_engine_type: str = "template"):
        """
        Initialize the solver worker.

        Args:
            process_name: Name of the game process to monitor
            ocr_engine_type: OCR engine type to use (default: "template")
        """
        super().__init__()
        self.process_name = process_name
        self._running = False
        self._capture: Optional[WindowCapture] = None

        # OCR engine
        self._ocr_engine = create_engine(ocr_engine_type)

        # FPS tracking
        self._frame_times: List[float] = []
        self._last_frame_start: float = 0.0

        # Debug image support
        self._last_frame: Optional[Image.Image] = None
        self._last_grid_info: Optional[GridInfo] = None
        self._last_ocr_result: Optional[OCRResult] = None
        self._solution_manager = SolutionManager()

        # Track last displayed move to avoid flickering during stabilization
        self._last_overlay_move: Optional[object] = None  # Move type

        # Cache window rect to prevent flashing from rect jitter between frames
        self._cached_window_rect: Optional[Tuple[int, int, int, int]] = None

        # Cache grid_info to prevent cell position jitter during stabilization
        self._cached_grid_info: Optional[GridInfo] = None

    def run(self):
        """
        Main worker loop. Called when thread starts.

        Continuously monitors for the game window and captures frames.
        Emits signals to update the UI with current status.
        Maintains 10 FPS cap with accurate timing.
        """
        self._running = True
        self._capture = WindowCapture(self.process_name)
        self._frame_times.clear()

        logger.info("Solver worker started")
        self.status_changed.emit("Running")

        while self._running:
            frame_start = time.perf_counter()

            try:
                self._process_cycle()
            except Exception as e:
                logger.exception("Error in worker cycle")
                self.error_occurred.emit(str(e))

            # Calculate frame time and FPS
            frame_time_ms = (time.perf_counter() - frame_start) * 1000
            self._update_fps(frame_time_ms)

            # Throttle to maintain FPS cap
            if frame_time_ms < self.MIN_FRAME_TIME_MS:
                sleep_time = int(self.MIN_FRAME_TIME_MS - frame_time_ms)
                self.msleep(sleep_time)

        # Cleanup
        self._capture.release()
        self._capture = None
        logger.info("Solver worker stopped")

    def _process_cycle(self):
        """
        Single iteration of the worker loop.

        Checks window status, captures frame if available,
        runs OCR processing, and emits status updates.
        """
        # Try to find window if not currently tracking
        if not self._capture.window_info:
            if self._capture.find_window():
                logger.info(f"Found window: {self._capture.window_info.title}")
                self.window_changed.emit(self._capture.get_status_string())
            else:
                self.window_changed.emit("Not detected")
                self.board_changed.emit("--")
                self.moves_changed.emit(0, "--", 0)
                self.ocr_status.emit(0.0, 0, 0)
                return

        # Check if window is still valid
        if not self._capture.is_active():
            logger.info("Window lost")
            self._capture.release()
            self._cached_window_rect = None
            self.window_changed.emit("Window lost - searching...")
            self.board_changed.emit("--")
            self.moves_changed.emit(0, "--", 0)
            self.ocr_status.emit(0.0, 0, 0)
            return

        # Update window info if moved/resized - also update cached rect
        if self._capture.has_moved():
            self._cached_window_rect = self._capture.get_rect()
            self.window_changed.emit(self._capture.get_status_string())

        # Initialize cached rect if needed
        if self._cached_window_rect is None:
            self._cached_window_rect = self._capture.get_rect()

        # Capture frame
        frame = self._capture.grab_frame()
        if frame:
            # Store frame for debug export
            self._last_frame = frame

            # Phase 4 - OCR processing using pluggable engine
            ocr_result = self._ocr_engine.process(frame)
            grid_info = ocr_result.grid_info

            if grid_info:
                self._last_grid_info = grid_info
                self._last_ocr_result = ocr_result

                # Emit board status
                self.board_changed.emit(f"{grid_info.rows}x{grid_info.cols} detected")

                # Emit OCR status (confidence as percentage)
                self.ocr_status.emit(
                    ocr_result.confidence * 100,
                    ocr_result.total_cells,
                    ocr_result.uncertain_count
                )

                # Phase 5 - Solve board (single-move-at-a-time)
                board_state = BoardState.from_ocr(ocr_result.board)
                has_move = self._solution_manager.update(board_state)
                next_move = self._solution_manager.current_move
                current_state = self._solution_manager.state
                self.next_move_ready.emit(next_move)

                # Phase 6 - Emit overlay update with move, grid_info, and window rect
                # Key: Use cached window_rect to prevent flashing from rect jitter
                # Key: Use cached grid_info to prevent cell position jitter

                # Get current move as list for overlay (single move or empty)
                current_moves = [next_move] if next_move else []

                if next_move:
                    # Have a move to display
                    if next_move != self._last_overlay_move:
                        # Log when move changes
                        logger.info(f"Next move: rect({next_move.r1},{next_move.c1})->({next_move.r2},{next_move.c2}), {len(next_move.cells)} cells, sum={next_move.total}")

                        # Log the OCR grid matrix for debugging coordinate issues
                        logger.info("OCR Grid Matrix (row, col) - rows are top to bottom, cols are left to right:")
                        for row_idx, row in enumerate(ocr_result.board):
                            row_str = " ".join(str(v) if v is not None else "." for v in row)
                            logger.info(f"  Row {row_idx:2d}: [{row_str}]")

                        # Log the move cells with their values
                        cell_values = []
                        for r, c in next_move.cells:
                            val = ocr_result.board[r][c] if 0 <= r < len(ocr_result.board) and 0 <= c < len(ocr_result.board[0]) else None
                            cell_values.append(f"({r},{c})={val}")
                        logger.info(f"Move cells: {', '.join(cell_values)}")
                    self._last_overlay_move = next_move

                    # Cache grid_info when we have a stable move
                    if current_state == SolutionState.MOVE_DISPLAYED:
                        self._cached_grid_info = grid_info

                    # Use cached grid_info if available, otherwise current
                    display_grid_info = self._cached_grid_info if self._cached_grid_info else grid_info
                    self.overlay_update.emit(next_move, display_grid_info, self._cached_window_rect, frame, current_moves, current_state)

                    # Build status message based on state
                    state_str = self._solution_manager.get_state_string()
                    expected_count = len(self._solution_manager.expected_cleared)

                    if current_state == SolutionState.MOVE_DISPLAYED:
                        status = f"Ready ({next_move.cell_count} cells)"
                    elif current_state == SolutionState.VALIDATING_CLEAR:
                        status = f"Validating ({expected_count} cells)"
                    else:
                        status = f"{state_str}"

                    # Emit solution stats: total_moves, status, total_cells
                    total_moves = self._solution_manager.total_moves
                    total_cells = self._solution_manager.total_cleared
                    self.moves_changed.emit(total_moves, status, total_cells)
                else:
                    # No move to display
                    self._last_overlay_move = None
                    self._cached_grid_info = None
                    self.overlay_update.emit(None, None, None, None, None, None)

                    # Status based on state
                    if current_state == SolutionState.WAITING_STABLE:
                        self.moves_changed.emit(0, "Stabilizing...", 0)
                    elif current_state == SolutionState.COMPUTING_MOVE:
                        self.moves_changed.emit(0, "Computing...", 0)
                    else:
                        self.moves_changed.emit(0, "No valid moves", 0)
            else:
                self._last_grid_info = None
                self._last_ocr_result = None
                self.board_changed.emit("Grid not detected")
                self.ocr_status.emit(0.0, 0, 0)
                self.moves_changed.emit(0, "--", 0)
                self.overlay_update.emit(None, None, None, None, None, None)
        else:
            self.board_changed.emit("Capture failed")
            self.ocr_status.emit(0.0, 0, 0)
            self.overlay_update.emit(None, None, None, None, None, None)

    def set_strategy(self, strategy_name: str):
        """
        Change the solving strategy.

        Args:
            strategy_name: Name of strategy to use (e.g., "greedy", "greedy_plus")
        """
        logger.info(f"Strategy change requested: {strategy_name}")
        self._solution_manager.set_strategy(strategy_name)

    def request_stop(self):
        """
        Request the worker to stop gracefully.

        The worker will complete its current cycle before stopping.
        Use wait() after calling this to block until stopped.
        """
        logger.info("Stop requested")
        self._running = False
        self._solution_manager.reset()
        self._last_overlay_move = None
        self._cached_window_rect = None
        self._cached_grid_info = None

    def is_running(self) -> bool:
        """
        Check if the worker is currently running.

        Returns:
            True if worker loop is active, False otherwise
        """
        return self._running

    def _update_fps(self, frame_time_ms: float) -> None:
        """
        Update FPS rolling average and emit signal.

        Args:
            frame_time_ms: Time taken for current frame in milliseconds
        """
        # Add current frame time to window
        self._frame_times.append(frame_time_ms)

        # Keep only last N frames
        if len(self._frame_times) > self.FPS_WINDOW_SIZE:
            self._frame_times.pop(0)

        # Calculate average FPS
        if self._frame_times:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            current_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        else:
            current_fps = 0.0

        # Emit FPS update
        self.fps_update.emit(current_fps, self.FPS_CAP)

    def save_debug_image(self) -> Optional[str]:
        """
        Save current frame with OCR annotations to debug directory.

        Returns:
            Path to saved file, or None if no frame available
        """
        if self._last_frame is None:
            logger.warning("No frame available for debug image")
            return None

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"debug_{timestamp}.png"
        filepath = DEBUG_DIR / filename

        # Save annotated image
        save_debug_image(
            self._last_frame,
            self._last_grid_info,
            self._last_ocr_result,
            str(filepath)
        )

        logger.info(f"Debug image saved: {filepath}")
        return str(filepath)
