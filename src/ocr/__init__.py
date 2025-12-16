"""
OCR Module for Nikke Math Solver

Pluggable OCR architecture for extracting game board values from images.

Usage:
    from src.ocr import create_engine, OCRResult

    # Create an OCR engine (template matching)
    engine = create_engine()

    # Process an image
    result = engine.process(image)

    # Access the board (2D array of digits 1-9 or None)
    board = result.board

Example with custom templates:
    engine = create_engine("template", template_dir="./my_templates")
"""

# Public API - Result types
from .result import (
    GridInfo,
    CellResult,
    OCRResult,
)

# Public API - Base class for custom engines
from .base import OCREngine

# Public API - Factory functions
from .factory import (
    create_engine,
    register_engine,
    available_engines,
)

# Public API - Template engine
from .template_engine import TemplateOCREngine

# Constants re-exported for backward compatibility
from .template_engine import (
    EXPECTED_ROWS,
    EXPECTED_COLS,
    CONFIDENCE_THRESHOLD,
    preprocess_cell,
    detect_grid,
)

# Debug utilities
from .debug import DEBUG_DIR, save_debug_image

__all__ = [
    # Result types
    "GridInfo",
    "CellResult",
    "OCRResult",
    # Base class
    "OCREngine",
    # Factory
    "create_engine",
    "register_engine",
    "available_engines",
    # Engines
    "TemplateOCREngine",
    # Constants
    "EXPECTED_ROWS",
    "EXPECTED_COLS",
    "CONFIDENCE_THRESHOLD",
    # Functions
    "preprocess_cell",
    "detect_grid",
    "save_debug_image",
    # Debug
    "DEBUG_DIR",
]
