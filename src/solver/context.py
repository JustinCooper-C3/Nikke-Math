"""
Solution Context Module - Shared context for strategy execution.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .board import BoardState


@dataclass
class SolutionContext:
    """
    Shared context passed to strategies containing board state,
    cancellation, and progress reporting.

    Attributes:
        board: Current board state to solve
        cancel_flag: Threading event for cancellation
        timeout_sec: Maximum computation time in seconds
        start_time: When computation started
        progress_callback: Optional callback for progress updates
    """
    board: BoardState
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    timeout_sec: float = 20.0
    start_time: float = field(default_factory=time.time)
    progress_callback: Optional[Callable[[float, str], None]] = None

    def is_cancelled(self) -> bool:
        """
        Check if cancellation requested or timeout exceeded.

        Returns:
            True if strategy should stop execution
        """
        if self.cancel_flag.is_set():
            return True
        if time.time() - self.start_time > self.timeout_sec:
            return True
        return False

    def report_progress(self, percent: float, message: str = "") -> None:
        """
        Report progress to UI.

        Args:
            percent: Progress from 0.0 to 1.0
            message: Optional status message
        """
        if self.progress_callback:
            self.progress_callback(percent, message)

    def elapsed_time(self) -> float:
        """
        Get seconds elapsed since computation started.

        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time

    def remaining_time(self) -> float:
        """
        Get seconds remaining before timeout.

        Returns:
            Remaining time in seconds (may be negative if exceeded)
        """
        return self.timeout_sec - self.elapsed_time()
