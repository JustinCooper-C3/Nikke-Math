"""
OCR Result Dataclasses

Shared data structures for OCR engine results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GridInfo:
    """Grid detection results."""
    bounds: Tuple[int, int, int, int]  # (x, y, width, height) of grid area
    rows: int
    cols: int
    cell_size: int  # Estimated cell diameter
    cell_gap: int   # Gap between cells
    cell_positions: List[List[Tuple[int, int]]] = field(default_factory=list)  # [row][col] = (x, y)
    # Origin point for stable coordinate mapping (top-left of column 0, row 0)
    origin: Optional[Tuple[int, int]] = None
    # Cell stride for calculating positions of missing cells
    cell_stride: int = 0


@dataclass
class CellResult:
    """Per-cell OCR result."""
    row: int
    col: int
    value: Optional[int]  # 1-9 or None if uncertain
    confidence: float     # 0.0-1.0
    position: Tuple[int, int]  # (x, y) center coordinates
    raw_scores: List[float] = field(default_factory=list)  # Match scores for each digit


@dataclass
class OCRResult:
    """Complete OCR result for a frame."""
    board: List[List[Optional[int]]]  # 2D array [row][col] of values
    confidence: float                  # Average confidence
    cell_results: List[CellResult]     # Per-cell details
    grid_info: Optional[GridInfo]      # Grid bounds and positions
    uncertain_count: int               # Cells with confidence < threshold
    total_cells: int                   # Total cells detected
    processing_time_ms: float          # Time taken
