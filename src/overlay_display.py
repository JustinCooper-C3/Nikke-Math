"""
Overlay Display Module for Nikke Math Solver

Provides transparent overlay rendering for displaying move hints over the game window.
Uses native PyQt5 transparent window instead of PythonOverlayLib to avoid threading issues.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

from PIL import Image, ImageDraw

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush

from src.solver import Move
from src.solution_manager import SolutionState
from src.ocr import GridInfo, DEBUG_DIR

# Configure module logger
logger = logging.getLogger(__name__)


# Color constants - Move Ready (green)
HIGHLIGHT_FILL = QColor(0, 255, 0, 80)       # Semi-transparent green fill
HIGHLIGHT_BORDER = QColor(0, 255, 0, 220)    # Bright green border

# Color constants - Validating (yellow/orange)
VALIDATING_FILL = QColor(255, 200, 0, 80)    # Semi-transparent yellow fill
VALIDATING_BORDER = QColor(255, 200, 0, 220) # Bright yellow border

BORDER_THICKNESS = 3                          # Border width in pixels
REFRESH_INTERVAL_MS = 50                      # 20 FPS update rate

# Flag for availability (always True with PyQt5)
OVERLAY_AVAILABLE = True


def grid_to_screen(
    row: int,
    col: int,
    grid_info: GridInfo,
    window_rect: Tuple[int, int, int, int],
    use_fallback_for_missing: bool = False
) -> Optional[Tuple[int, int]]:
    """
    Convert grid cell coordinates to screen pixel position.

    Args:
        row: Grid row index
        col: Grid column index
        grid_info: GridInfo with cell_positions
        window_rect: Game window (x, y, width, height) in screen coords
        use_fallback_for_missing: If True, calculate position for cleared cells

    Returns:
        (x, y) screen coordinates for cell center, or None if cell is missing
    """
    # Get cell center from grid_info (relative to captured image)
    if (grid_info.cell_positions and
        row < len(grid_info.cell_positions) and
        col < len(grid_info.cell_positions[row])):
        cell_x, cell_y = grid_info.cell_positions[row][col]
        # Check for missing cell marker
        if cell_x < 0 or cell_y < 0:
            if not use_fallback_for_missing:
                return None
            # Use origin-based fallback calculation for cleared/missing cells
            # This ensures consistent positioning even when cells are cleared
            if grid_info.origin and grid_info.cell_stride > 0:
                origin_x, origin_y = grid_info.origin
                cell_x = origin_x + col * grid_info.cell_stride
                cell_y = origin_y + row * grid_info.cell_stride
            else:
                # Legacy fallback using bounds (less accurate)
                gx, gy, gw, gh = grid_info.bounds
                cell_stride = grid_info.cell_size + grid_info.cell_gap
                cell_x = gx + col * cell_stride + grid_info.cell_size // 2
                cell_y = gy + row * cell_stride + grid_info.cell_size // 2
    else:
        # Fallback: calculate from origin if available
        if grid_info.origin and grid_info.cell_stride > 0:
            origin_x, origin_y = grid_info.origin
            cell_x = origin_x + col * grid_info.cell_stride
            cell_y = origin_y + row * grid_info.cell_stride
        else:
            # Legacy fallback using bounds
            gx, gy, gw, gh = grid_info.bounds
            cell_stride = grid_info.cell_size + grid_info.cell_gap
            cell_x = gx + col * cell_stride + grid_info.cell_size // 2
            cell_y = gy + row * cell_stride + grid_info.cell_size // 2

    # Add window offset to get screen coordinates
    screen_x = window_rect[0] + cell_x
    screen_y = window_rect[1] + cell_y

    return screen_x, screen_y


def get_move_screen_rect(
    move: Move,
    grid_info: GridInfo,
    window_rect: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculate screen rectangle for a move's selection area.

    Uses tight_bounds (bounding box of actual non-zero cells) instead of
    the full rectangle bounds to avoid including empty cells in the selection.

    Uses fallback calculation for cleared/missing cells so moves can
    still be drawn even after some cells have been cleared from the board.

    Args:
        move: Move with tight_bounds calculated from actual cells
        grid_info: GridInfo with cell positions and sizes
        window_rect: Game window position in screen coords

    Returns:
        (x, y, width, height) screen rectangle for the move, or None if calculation fails
    """
    # Use tight bounds (bounding box of actual cells) instead of full rectangle
    t_r1, t_c1, t_r2, t_c2 = move.tight_bounds

    # Get top-left cell center (use fallback for missing cells)
    top_left = grid_to_screen(t_r1, t_c1, grid_info, window_rect, use_fallback_for_missing=True)
    if top_left is None:
        return None
    top_left_x, top_left_y = top_left

    # Get bottom-right cell center (use fallback for missing cells)
    bottom_right = grid_to_screen(t_r2, t_c2, grid_info, window_rect, use_fallback_for_missing=True)
    if bottom_right is None:
        return None
    bottom_right_x, bottom_right_y = bottom_right

    # Expand from centers to full cell bounds
    half_cell = grid_info.cell_size // 2

    x = top_left_x - half_cell
    y = top_left_y - half_cell
    width = (bottom_right_x - top_left_x) + grid_info.cell_size
    height = (bottom_right_y - top_left_y) + grid_info.cell_size

    return x, y, width, height


class OverlayWindow(QWidget):
    """
    Transparent, click-through overlay window for displaying move hints.

    This window covers the entire screen and is:
    - Frameless (no title bar)
    - Transparent background
    - Click-through (mouse events pass through)
    - Always on top

    Supports two visual states:
    - Green: Move ready to execute
    - Yellow: Validating clear (waiting for cells to disappear)
    """

    def __init__(self):
        super().__init__()

        # Current rectangle to draw (None = nothing)
        self._rect: Optional[Tuple[int, int, int, int]] = None
        self._is_validating: bool = False  # Yellow state when True
        self._lock = threading.Lock()

        self._setup_window()

    def _setup_window(self):
        """Configure window properties for transparent overlay."""
        # Frameless, always on top, tool window (no taskbar icon)
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.WindowTransparentForInput  # Click-through
        )

        # Transparent background
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # Cover entire screen
        screen = QApplication.primaryScreen()
        if screen:
            geometry = screen.geometry()
            self.setGeometry(geometry)

    def set_rect(self, rect: Optional[Tuple[int, int, int, int]], is_validating: bool = False):
        """
        Set the rectangle to highlight.

        Args:
            rect: (x, y, width, height) or None to clear
            is_validating: If True, use yellow color to indicate validation state
        """
        with self._lock:
            self._rect = rect
            self._is_validating = is_validating
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        """Paint the highlight rectangle."""
        with self._lock:
            rect = self._rect
            is_validating = self._is_validating

        if rect is None:
            return

        x, y, width, height = rect

        # Choose colors based on state
        if is_validating:
            fill_color = VALIDATING_FILL
            border_color = VALIDATING_BORDER
        else:
            fill_color = HIGHLIGHT_FILL
            border_color = HIGHLIGHT_BORDER

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw filled rectangle
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.NoPen)
        painter.drawRect(x, y, width, height)

        # Draw border
        pen = QPen(border_color)
        pen.setWidth(BORDER_THICKNESS)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(x, y, width, height)

        painter.end()


class OverlayManager(QObject):
    """
    Manages the transparent overlay for displaying move hints.

    Thread-safe: update_move() can be called from worker thread.
    Must be created from main thread (PyQt5 requirement).

    Usage:
        manager = OverlayManager()
        manager.start()

        # From worker thread:
        manager.update_move(move, grid_info, window_rect, frame, solution_stack, solution_state)

        # When done:
        manager.stop()
    """

    # Signal for thread-safe updates from worker
    # (Move, GridInfo, window_rect, frame, solution_stack, solution_state)
    _update_signal = pyqtSignal(object, object, object, object, object, object)

    # Debug screenshot throttle interval (seconds)
    DEBUG_SCREENSHOT_INTERVAL = 1.0

    def __init__(self, debug_mode: bool = False):
        """
        Initialize overlay manager (does not start overlay).

        Args:
            debug_mode: If True, save debug screenshots on each overlay render
        """
        super().__init__()

        self._overlay: Optional[OverlayWindow] = None
        self._running = False
        self._debug_mode = debug_mode
        self._last_debug_save_time = 0.0

        # Connect internal signal for thread-safe updates
        self._update_signal.connect(self._on_update)

    def start(self) -> bool:
        """
        Start the overlay.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        self._overlay = OverlayWindow()
        self._overlay.show()
        self._running = True
        return True

    def stop(self):
        """Stop the overlay and cleanup."""
        self._running = False

        if self._overlay:
            self._overlay.hide()
            self._overlay.close()
            self._overlay = None

    def update_move(
        self,
        move: Optional[Move],
        grid_info: Optional[GridInfo],
        window_rect: Optional[Tuple[int, int, int, int]],
        frame: Optional[Image.Image] = None,
        solution_stack: Optional[List[Move]] = None,
        solution_state: Optional[SolutionState] = None
    ):
        """
        Update the move to display (thread-safe).

        Args:
            move: Move to highlight, or None to clear
            grid_info: Current grid info for coordinate mapping
            window_rect: Game window screen position (x, y, w, h)
            frame: Current captured frame (for debug screenshots)
            solution_stack: Full solution stack (for debug logging)
            solution_state: Current solution state (for color indication)
        """
        # Emit signal for thread-safe update on main thread
        self._update_signal.emit(move, grid_info, window_rect, frame, solution_stack, solution_state)

    def _on_update(self, move, grid_info, window_rect, frame=None, solution_stack=None, solution_state=None):
        """Handle update on main thread."""
        if not self._overlay or not self._running:
            logger.debug("Overlay update ignored: overlay not running")
            return

        if move is None or grid_info is None or window_rect is None:
            logger.debug("Overlay: clearing rect (no move/grid_info/window_rect)")
            self._overlay.set_rect(None)
            return

        # Determine if we're in validating state (show yellow)
        is_validating = (solution_state == SolutionState.VALIDATING_CLEAR)

        try:
            rect = get_move_screen_rect(move, grid_info, window_rect)
            if rect is None:
                logger.debug(f"Overlay: clearing rect (move cells not found in grid)")
                self._overlay.set_rect(None)
                return

            t_r1, t_c1, t_r2, t_c2 = move.tight_bounds
            logger.debug(f"Overlay: drawing rect at ({rect[0]}, {rect[1]}) size {rect[2]}x{rect[3]} for tight_bounds ({t_r1},{t_c1})->({t_r2},{t_c2}) validating={is_validating}")
            self._overlay.set_rect(rect, is_validating=is_validating)

            # Save debug screenshot if enabled and throttle interval passed
            if self._debug_mode and frame is not None:
                self._save_debug_screenshot(frame, move, grid_info, rect, solution_stack)

        except Exception as e:
            logger.warning(f"Overlay: failed to calculate rect: {e}")
            self._overlay.set_rect(None)

    def _save_debug_screenshot(
        self,
        frame: Image.Image,
        move: Move,
        grid_info: GridInfo,
        screen_rect: Tuple[int, int, int, int],
        solution_stack: Optional[List[Move]]
    ):
        """
        Save TWO debug screenshots: one clean (no overlay) and one with overlay rect drawn.

        Throttled to avoid flooding with screenshots.

        Args:
            frame: Captured game frame
            move: Current move being displayed
            grid_info: Grid detection info
            screen_rect: Screen coordinates of overlay rect
            solution_stack: Full solution for logging
        """
        current_time = time.time()
        if current_time - self._last_debug_save_time < self.DEBUG_SCREENSHOT_INTERVAL:
            return

        self._last_debug_save_time = current_time

        try:
            # Ensure debug directory exists
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # Save CLEAN frame first (no overlay drawn)
            clean_filename = f"debug_{timestamp}_clean.png"
            clean_filepath = DEBUG_DIR / clean_filename
            frame.save(str(clean_filepath), "PNG")

            # Create copy and draw the overlay rect on it
            debug_img = frame.copy()
            draw = ImageDraw.Draw(debug_img)

            # Calculate rect position relative to captured image
            # The move's grid position maps to image coordinates via grid_info
            half_cell = grid_info.cell_size // 2

            # Use tight bounds for debug drawing (matches actual overlay)
            t_r1, t_c1, t_r2, t_c2 = move.tight_bounds

            # Get top-left and bottom-right cell centers from grid_info
            if (grid_info.cell_positions and
                t_r1 < len(grid_info.cell_positions) and
                t_c1 < len(grid_info.cell_positions[t_r1]) and
                t_r2 < len(grid_info.cell_positions) and
                t_c2 < len(grid_info.cell_positions[t_r2])):

                tl_x, tl_y = grid_info.cell_positions[t_r1][t_c1]
                br_x, br_y = grid_info.cell_positions[t_r2][t_c2]

                img_x1 = tl_x - half_cell
                img_y1 = tl_y - half_cell
                img_x2 = br_x + half_cell
                img_y2 = br_y + half_cell

                # Draw green rectangle matching overlay
                draw.rectangle(
                    [img_x1, img_y1, img_x2, img_y2],
                    outline=(0, 255, 0),
                    width=3
                )

            # Save the image WITH overlay rect drawn
            overlay_filename = f"debug_{timestamp}_overlay.png"
            overlay_filepath = DEBUG_DIR / overlay_filename
            debug_img.save(str(overlay_filepath), "PNG")

            # Log solution details
            solution_info = ""
            if solution_stack:
                solution_info = f", Solution: {len(solution_stack)} moves"
                # Log each move in the solution
                moves_summary = []
                for i, m in enumerate(solution_stack[:5]):  # First 5 moves
                    moves_summary.append(f"  Move {i+1}: ({m.r1},{m.c1})->({m.r2},{m.c2}) {len(m.cells)} cells")
                if len(solution_stack) > 5:
                    moves_summary.append(f"  ... and {len(solution_stack) - 5} more moves")

                logger.info(
                    f"Debug screenshots saved:\n"
                    f"  Clean: {clean_filepath}\n"
                    f"  Overlay: {overlay_filepath}\n"
                    f"  Current move: ({move.r1},{move.c1})->({move.r2},{move.c2}), "
                    f"{len(move.cells)} cells, sum={move.total}\n"
                    f"  Screen rect: ({screen_rect[0]}, {screen_rect[1]}) "
                    f"size {screen_rect[2]}x{screen_rect[3]}\n"
                    f"  Grid: {grid_info.rows}x{grid_info.cols}, cell_size={grid_info.cell_size}\n"
                    f"  Solution ({len(solution_stack)} moves):\n" +
                    "\n".join(moves_summary)
                )
            else:
                logger.info(
                    f"Debug screenshots saved:\n"
                    f"  Clean: {clean_filepath}\n"
                    f"  Overlay: {overlay_filepath}\n"
                    f"  Current move: ({move.r1},{move.c1})->({move.r2},{move.c2}), "
                    f"{len(move.cells)} cells, sum={move.total}\n"
                    f"  Screen rect: ({screen_rect[0]}, {screen_rect[1]}) "
                    f"size {screen_rect[2]}x{screen_rect[3]}\n"
                    f"  Grid: {grid_info.rows}x{grid_info.cols}, cell_size={grid_info.cell_size}"
                )

        except Exception as e:
            logger.warning(f"Failed to save debug screenshot: {e}")

    def clear_move(self):
        """Clear the current move display."""
        self._update_signal.emit(None, None, None, None, None, None)

    @property
    def is_running(self) -> bool:
        """Check if overlay is currently running."""
        return self._running


# Singleton instance for easy access
_overlay_manager: Optional[OverlayManager] = None


def get_overlay_manager(debug_mode: bool = False) -> OverlayManager:
    """
    Get the singleton OverlayManager instance.

    Args:
        debug_mode: Enable debug screenshots (only used on first call)

    Returns:
        OverlayManager instance (created if needed)
    """
    global _overlay_manager
    if _overlay_manager is None:
        _overlay_manager = OverlayManager(debug_mode=debug_mode)
    return _overlay_manager
