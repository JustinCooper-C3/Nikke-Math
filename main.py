"""
Nikke Math Solver - Entry Point

Launches the Control UI window and manages the solver worker thread.

Example:
    python main.py
    python main.py --process notepad.exe  # Test with different process
"""

import sys
import logging
import argparse
from typing import Optional

from PyQt5.QtWidgets import QApplication

from src.control_ui import ControlWindow
from src.solver_worker import SolverWorker
from src.overlay_display import OverlayManager, OVERLAY_AVAILABLE
from src.settings import load_settings, save_settings


# Configure logging - output to both console and file
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level to see stability checks
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("solver.log", mode='w', encoding='utf-8')  # File output
    ]
)
logger = logging.getLogger(__name__)


class Application:
    """
    Main application controller.

    Manages the lifecycle of the UI and worker thread,
    connecting signals between them.
    """

    def __init__(self, process_name: str = "nikke.exe", debug_mode: bool = False):
        """
        Initialize the application.

        Args:
            process_name: Name of the game process to monitor
            debug_mode: Enable debug mode via CLI (overrides saved setting)
        """
        self.process_name = process_name
        self.cli_debug_override = debug_mode  # CLI flag overrides saved setting
        self.window: Optional[ControlWindow] = None
        self.worker: Optional[SolverWorker] = None
        self.overlay: Optional[OverlayManager] = None

        # Load persistent settings
        self.settings = load_settings()

        # Effective debug mode: CLI flag overrides saved setting
        if self.cli_debug_override:
            self.debug_mode = True
        else:
            self.debug_mode = self.settings.get("debug_enabled", False)

    def setup(self):
        """Set up the UI and connect signals."""
        # Create UI window
        self.window = ControlWindow()

        # Connect UI signals to handlers
        self.window.start_requested.connect(self._on_start)
        self.window.stop_requested.connect(self._on_stop)
        self.window.shutdown_requested.connect(self._on_shutdown)
        self.window.strategy_changed.connect(self._on_strategy_changed)
        self.window.debug_toggled.connect(self._on_debug_toggled)

        # Initialize UI state from settings
        self.window.set_debug_enabled(self.debug_mode)
        self._set_strategy_from_settings()

        # Create overlay manager with debug mode
        self.overlay = OverlayManager(debug_mode=self.debug_mode)
        if OVERLAY_AVAILABLE:
            logger.info("Overlay system available")
        else:
            logger.warning("PythonOverlayLib not installed - overlay disabled")

        if self.debug_mode:
            logger.info("Debug mode enabled - screenshots will be saved on overlay render")

        logger.info(f"Application initialized, monitoring: {self.process_name}")

    def _on_start(self):
        """Handle start button click."""
        if self.worker and self.worker.isRunning():
            logger.warning("Worker already running")
            return

        logger.info("Starting solver worker")

        # Create and configure worker
        self.worker = SolverWorker(self.process_name)

        # Set initial strategy from UI dropdown
        initial_strategy = self.window.strategy_combo.currentData()
        if initial_strategy:
            self.worker.set_strategy(initial_strategy)

        # Connect worker signals to UI
        self.worker.status_changed.connect(self.window.set_status)
        self.worker.window_changed.connect(self.window.set_window_info)
        self.worker.board_changed.connect(self.window.set_board_info)
        self.worker.moves_changed.connect(self.window.set_moves_info)
        self.worker.error_occurred.connect(self._on_error)

        # Phase 4 signals
        self.worker.ocr_status.connect(self._on_ocr_status)
        self.worker.fps_update.connect(self.window.set_fps_info)

        # Phase 6 - Overlay signals
        self.worker.overlay_update.connect(self._on_overlay_update)

        # Connect UI debug request to worker
        self.window.debug_requested.connect(self._on_debug_requested)

        # Start worker thread
        self.worker.start()

        # Start overlay
        if self.overlay and OVERLAY_AVAILABLE:
            if self.overlay.start():
                logger.info("Overlay started")
            else:
                logger.warning("Failed to start overlay")

        # Update UI state
        self.window.set_running(True)

    def _on_stop(self):
        """Handle stop button click."""
        if not self.worker or not self.worker.isRunning():
            logger.warning("Worker not running")
            return

        logger.info("Stopping solver worker")

        # Stop overlay first
        if self.overlay:
            self.overlay.stop()
            logger.info("Overlay stopped")

        # Request stop and wait
        self.worker.request_stop()
        self.worker.wait(2000)  # 2 second timeout

        if self.worker.isRunning():
            logger.warning("Worker did not stop gracefully, terminating")
            self.worker.terminate()
            self.worker.wait()

        self.worker = None

        # Update UI state
        self.window.set_running(False)
        self.window.set_window_info("Not detected")
        self.window.set_board_info("--")
        self.window.set_moves_info("--")

    def _on_shutdown(self):
        """Handle window close."""
        logger.info("Shutdown requested")
        self._on_stop()

    def _on_error(self, error_msg: str):
        """Handle worker error."""
        logger.error(f"Worker error: {error_msg}")
        self.window.set_status(f"Error: {error_msg}")

    def _on_ocr_status(self, confidence: float, total: int, uncertain: int):
        """Handle OCR status update from worker."""
        self.window.set_ocr_info(confidence)
        self.window.set_cells_info(total, uncertain)

    def _on_strategy_changed(self, strategy_name: str):
        """Handle strategy selection change from UI."""
        logger.info(f"Strategy changed to: {strategy_name}")
        if self.worker and self.worker.isRunning():
            self.worker.set_strategy(strategy_name)

        # Save to persistent settings
        self.settings["strategy_name"] = strategy_name
        save_settings(self.settings)

    def _on_debug_toggled(self, enabled: bool):
        """Handle debug checkbox toggle from UI."""
        logger.info(f"Debug mode toggled: {enabled}")
        self.debug_mode = enabled

        # Update overlay manager
        if self.overlay:
            self.overlay.set_debug_mode(enabled)

        # Save to persistent settings (only if not CLI override)
        if not self.cli_debug_override:
            self.settings["debug_enabled"] = enabled
            save_settings(self.settings)

    def _set_strategy_from_settings(self):
        """Set strategy dropdown from saved settings."""
        saved_strategy = self.settings.get("strategy_name")
        if saved_strategy:
            # Find and select the saved strategy in the dropdown
            for i in range(self.window.strategy_combo.count()):
                if self.window.strategy_combo.itemData(i) == saved_strategy:
                    self.window.strategy_combo.setCurrentIndex(i)
                    logger.debug(f"Restored strategy from settings: {saved_strategy}")
                    return
            logger.debug(f"Saved strategy '{saved_strategy}' not found, using default")

    def _on_overlay_update(self, move, grid_info, window_rect, frame, solution_stack, solution_state):
        """Handle overlay update from worker."""
        if self.overlay:
            self.overlay.update_move(move, grid_info, window_rect, frame, solution_stack, solution_state)

    def _on_debug_requested(self):
        """Handle debug image save request."""
        if self.worker and self.worker.isRunning():
            path = self.worker.save_debug_image()
            if path:
                logger.info(f"Debug image saved: {path}")
            else:
                logger.warning("Failed to save debug image")

    def run(self) -> int:
        """
        Run the application.

        Returns:
            Exit code
        """
        self.window.show()
        return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nikke Math Solver - Puzzle solver with overlay display"
    )
    parser.add_argument(
        "--process", "-p",
        default="nikke.exe",
        help="Process name to monitor (default: nikke.exe)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode (auto-save screenshots on overlay render)"
    )
    return parser.parse_args()


def main():
    """Initialize and run the Nikke Math Solver application."""
    args = parse_args()

    app = QApplication(sys.argv)

    application = Application(process_name=args.process, debug_mode=args.debug)
    application.setup()
    application.run()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
