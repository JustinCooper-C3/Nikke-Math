"""
Control UI Module for Nikke Math Solver

Provides a PyQt5-based control window for managing the puzzle solver application.
Includes start/stop controls, status display, and worker thread communication signals.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from src.ocr.debug import get_confidence_color
from src.solver import get_strategy_info


class ControlWindow(QMainWindow):
    """
    Main control window for the Nikke Math Solver application.

    Provides UI controls for starting/stopping the solver and displays
    real-time status information about window detection, board state, and moves.
    """

    # Signals for worker thread communication
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    shutdown_requested = pyqtSignal()
    debug_requested = pyqtSignal()  # Request debug image save
    strategy_changed = pyqtSignal(str)  # Emits strategy name when changed

    def __init__(self):
        super().__init__()
        self._is_running = False
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface components."""
        # Window configuration
        self.setWindowTitle("Nikke Math Solver")
        self.setFixedSize(320, 360)  # Increased height for new labels
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(layout)

        # Status label
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(10)
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        layout.addWidget(self.status_label)

        # Spacing
        layout.addSpacing(5)

        # Strategy selector
        strategy_layout = QHBoxLayout()
        strategy_label = QLabel("Strategy:")
        strategy_label.setFont(QFont("", 9))
        strategy_layout.addWidget(strategy_label)

        self.strategy_combo = QComboBox()
        for info in get_strategy_info():
            self.strategy_combo.addItem(info["description"], info["name"])
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo, 1)  # stretch factor 1
        layout.addLayout(strategy_layout)

        # Spacing
        layout.addSpacing(5)

        # Start/Stop button
        self.toggle_button = QPushButton("START SOLVER")
        self.toggle_button.setMinimumHeight(50)
        button_font = QFont()
        button_font.setPointSize(11)
        button_font.setBold(True)
        self.toggle_button.setFont(button_font)
        self.toggle_button.clicked.connect(self._on_toggle_clicked)
        layout.addWidget(self.toggle_button)

        # Spacing
        layout.addSpacing(10)

        # Info labels
        self.window_label = QLabel("Window: Not detected")
        self.board_label = QLabel("Board:  --")
        self.ocr_label = QLabel("OCR:    --")
        self.cells_label = QLabel("Cells:  --")
        self.fps_label = QLabel("FPS:    --")
        self.moves_label = QLabel("Moves:  --")

        info_font = QFont()
        info_font.setPointSize(9)

        for label in [self.window_label, self.board_label, self.ocr_label,
                      self.cells_label, self.fps_label, self.moves_label]:
            label.setFont(info_font)
            layout.addWidget(label)

        # Spacing before debug button
        layout.addSpacing(10)

        # Debug button
        self.debug_button = QPushButton("Save Debug Image")
        self.debug_button.setMinimumHeight(35)
        debug_font = QFont()
        debug_font.setPointSize(9)
        self.debug_button.setFont(debug_font)
        self.debug_button.clicked.connect(self._on_debug_clicked)
        self.debug_button.setEnabled(False)  # Disabled until running
        layout.addWidget(self.debug_button)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Apply styling
        self._apply_styles()

    def _apply_styles(self):
        """Apply clean, minimal styling to the window."""
        style = """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                color: #333333;
            }
        """
        self.setStyleSheet(style)

    def _on_toggle_clicked(self):
        """Handle Start/Stop button click."""
        if self._is_running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def _on_debug_clicked(self):
        """Handle Save Debug Image button click."""
        self.debug_requested.emit()

    def _on_strategy_changed(self, index: int):
        """Handle strategy dropdown selection change."""
        strategy_name = self.strategy_combo.itemData(index)
        if strategy_name:
            self.strategy_changed.emit(strategy_name)

    def set_status(self, status: str):
        """
        Update the status label.

        Args:
            status: Status text to display (e.g., "Stopped", "Running", "Error: message")
        """
        self.status_label.setText(f"Status: {status}")

        # Color coding for different statuses
        if status.lower().startswith("error"):
            self.status_label.setStyleSheet("color: #d32f2f;")
        elif status.lower() == "running":
            self.status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.status_label.setStyleSheet("color: #333333;")

    def set_window_info(self, info: str):
        """
        Update the window detection label.

        Args:
            info: Window status information
        """
        self.window_label.setText(f"Window: {info}")

    def set_board_info(self, info: str):
        """
        Update the board detection label.

        Args:
            info: Board status information
        """
        self.board_label.setText(f"Board:  {info}")

    def set_moves_info(self, info: str):
        """
        Update the moves count label.

        Args:
            info: Moves count information
        """
        self.moves_label.setText(f"Moves:  {info}")

    def set_ocr_info(self, confidence: float):
        """
        Update the OCR confidence label with color coding.

        Args:
            confidence: OCR confidence percentage (0-100)
        """
        if confidence > 0:
            color = get_confidence_color(confidence / 100.0)
            self.ocr_label.setText(f"OCR:    {confidence:.1f}% confidence")
            self.ocr_label.setStyleSheet(f"color: {color};")
        else:
            self.ocr_label.setText("OCR:    --")
            self.ocr_label.setStyleSheet("color: #333333;")

    def set_cells_info(self, total: int, uncertain: int):
        """
        Update the cell count label.

        Args:
            total: Total cells detected
            uncertain: Number of uncertain cells
        """
        if total > 0:
            self.cells_label.setText(f"Cells:  {total} read ({uncertain} uncertain)")
        else:
            self.cells_label.setText("Cells:  --")

    def set_fps_info(self, current: float, cap: float):
        """
        Update the FPS label.

        Args:
            current: Current FPS value
            cap: FPS cap value
        """
        self.fps_label.setText(f"FPS:    {current:.1f} (cap: {int(cap)})")

    def set_running(self, is_running: bool):
        """
        Toggle the button state and update status.

        Args:
            is_running: True if solver is running, False if stopped
        """
        self._is_running = is_running

        # Enable/disable buttons based on running state
        self.debug_button.setEnabled(is_running)
        self.strategy_combo.setEnabled(not is_running)  # Disable while running

        if is_running:
            self.toggle_button.setText("STOP SOLVER")
            self.toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:pressed {
                    background-color: #c41408;
                }
            """)
            self.set_status("Running")
        else:
            self.toggle_button.setText("START SOLVER")
            self.toggle_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            self.set_status("Stopped")

            # Reset info labels when stopped
            self.set_ocr_info(0)
            self.set_cells_info(0, 0)
            self.fps_label.setText("FPS:    --")

    def closeEvent(self, event):
        """
        Handle window close event.

        Emits shutdown_requested signal before closing to allow
        graceful cleanup of worker threads.

        Args:
            event: QCloseEvent object
        """
        self.shutdown_requested.emit()
        event.accept()
