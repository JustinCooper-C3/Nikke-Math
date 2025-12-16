"""
Template Matching OCR Engine

OCR implementation using OpenCV template matching for digit recognition.
Uses white threshold detection for high-accuracy grid recognition.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .base import OCREngine
from .result import GridInfo, CellResult, OCRResult


# Grid constants (10 columns x 16 rows)
EXPECTED_ROWS = 16
EXPECTED_COLS = 10
EXPECTED_CELLS = EXPECTED_ROWS * EXPECTED_COLS  # 160 cells

# Reference dimensions (calibration size for hardcoded pixel values)
# These are the image dimensions the grid bounds were calibrated for
# Based on debug/example_clean_grid.png (1796x1040) measurements
REFERENCE_WIDTH = 1796
REFERENCE_HEIGHT = 1040

# Cell detection parameters (tested configuration) - at reference scale
WHITE_THRESHOLD = 240  # Using pure white RGB detection now
MIN_CELL_AREA_REF = 15  # Lowered from 30 to catch green-tinted digits
MAX_CELL_AREA_REF = 800
MIN_CELL_WIDTH_REF = 5
MAX_CELL_WIDTH_REF = 50  # Increased for larger window sizes
MIN_CELL_HEIGHT_REF = 8
MAX_CELL_HEIGHT_REF = 50  # Increased for larger window sizes

# Grid region bounds (game board location) - at reference scale
# Measured from debug images: actual grid X range 648-1109, Y range 249-969
# Y=249 is first real grid row; rows 0-6 (Y<200) are UI header elements
GRID_X_MIN_REF = 600   # Slightly below 648 for margin
GRID_X_MAX_REF = 1150  # Slightly above 1109 for margin
GRID_Y_MIN_REF = 220   # Above header UI, below first grid row (~249)
GRID_Y_MAX_REF = 1000  # Above 969 for margin

# Row grouping threshold (pixels) - at reference scale
ROW_GROUPING_THRESHOLD_REF = 15

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7

# Empty cell validation parameters
# Variance threshold: low variance = uniform background = truly empty
# High variance = content present but failed detection
EMPTY_CELL_VARIANCE_THRESHOLD = 500  # Below this = truly empty cell
CELL_REGION_PADDING = 5  # Extra pixels around expected cell center


class DigitTemplates:
    """Manages digit templates for template matching."""

    def __init__(self):
        self.templates: dict[int, np.ndarray] = {}
        self._loaded = False

    def load_templates(self, template_dir: Path) -> bool:
        """
        Load digit templates from directory.

        Expected files: 1.png, 2.png, ... 9.png

        Args:
            template_dir: Path to directory containing template images

        Returns:
            True if templates loaded successfully
        """
        self.templates.clear()

        if not template_dir.exists():
            return False

        for digit in range(1, 10):
            template_path = template_dir / f"{digit}.png"
            if template_path.exists():
                img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[digit] = img

        self._loaded = len(self.templates) == 9
        return self._loaded

    def is_loaded(self) -> bool:
        """Check if templates are loaded."""
        return self._loaded

    def match(self, cell_image: np.ndarray) -> Tuple[Optional[int], float, List[float]]:
        """
        Match a cell image against digit templates.

        Args:
            cell_image: Grayscale/binary cell image

        Returns:
            Tuple of (best_digit, confidence, all_scores)
        """
        if not self._loaded:
            return None, 0.0, []

        scores = []

        for digit in range(1, 10):
            template = self.templates[digit]

            if template.shape != cell_image.shape:
                template = cv2.resize(template, (cell_image.shape[1], cell_image.shape[0]))

            result = cv2.matchTemplate(cell_image, template, cv2.TM_CCOEFF_NORMED)
            score = float(result[0, 0]) if result.size == 1 else float(np.max(result))
            scores.append(score)

        best_idx = int(np.argmax(scores))
        best_digit = best_idx + 1
        best_score = scores[best_idx]

        # Calculate confidence (convert [-1, 1] to [0, 1])
        confidence = max(0.0, min(1.0, (best_score + 1) / 2))

        if confidence < CONFIDENCE_THRESHOLD:
            return None, confidence, scores

        return best_digit, confidence, scores


class TemplateOCREngine(OCREngine):
    """
    OCR engine using OpenCV template matching.

    Detects game grid cells using white threshold detection
    and recognizes digits 1-9 using template matching.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the template OCR engine.

        Args:
            template_dir: Optional path to custom digit templates.
                         If None, uses ./assets/templates.
        """
        self._templates = DigitTemplates()
        self._template_dir = template_dir
        self._templates_loaded = False

    @property
    def name(self) -> str:
        return "template"

    def process(self, image: Image.Image) -> OCRResult:
        """
        Process an image and extract the game board.

        Args:
            image: PIL Image of the game window

        Returns:
            OCRResult with board values and confidence scores
        """
        start_time = time.perf_counter()

        # Ensure templates are loaded
        if not self._templates_loaded:
            self._load_templates()

        # Detect grid using white threshold
        grid_info, cell_images, original_image = self._detect_grid(image)

        if grid_info is None:
            return OCRResult(
                board=[],
                confidence=0.0,
                cell_results=[],
                grid_info=None,
                uncertain_count=0,
                total_cells=0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )

        # Extract board values (pass original image for empty cell validation)
        return self._extract_board(cell_images, grid_info, start_time, original_image)

    def configure(self, **kwargs) -> None:
        """
        Configure engine parameters.

        Args:
            template_dir: Path to custom digit templates
        """
        if 'template_dir' in kwargs:
            self._template_dir = Path(kwargs['template_dir'])
            self._templates_loaded = False

    def _load_templates(self) -> None:
        """Load digit templates from directory."""
        template_dir = self._template_dir or Path("./assets/templates")
        self._templates_loaded = self._templates.load_templates(template_dir)

    def _detect_grid(self, image: Image.Image) -> Tuple[Optional[GridInfo], List[List[np.ndarray]], Optional[np.ndarray]]:
        """
        Detect the game grid using white threshold detection.

        Builds a fixed-column grid (EXPECTED_COLS) where empty cells are
        represented with None images (which become 0 in the board).

        Returns:
            Tuple of (GridInfo, cell_images, original_bgr_image) or (None, [], None) if detection fails
        """
        # Convert to BGR for processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Calculate scale factors based on captured image size vs reference
        img_height, img_width = cv_image.shape[:2]
        scale_x = img_width / REFERENCE_WIDTH
        scale_y = img_height / REFERENCE_HEIGHT
        # Use average scale for area calculations
        scale_avg = (scale_x + scale_y) / 2

        # Scale grid bounds
        grid_x_min = int(GRID_X_MIN_REF * scale_x)
        grid_x_max = int(GRID_X_MAX_REF * scale_x)
        grid_y_min = int(GRID_Y_MIN_REF * scale_y)
        grid_y_max = int(GRID_Y_MAX_REF * scale_y)

        # Scale cell detection parameters
        min_cell_area = int(MIN_CELL_AREA_REF * scale_avg * scale_avg)
        max_cell_area = int(MAX_CELL_AREA_REF * scale_avg * scale_avg)
        min_cell_width = int(MIN_CELL_WIDTH_REF * scale_x)
        max_cell_width = int(MAX_CELL_WIDTH_REF * scale_x)
        min_cell_height = int(MIN_CELL_HEIGHT_REF * scale_y)
        max_cell_height = int(MAX_CELL_HEIGHT_REF * scale_y)
        row_grouping_threshold = int(ROW_GROUPING_THRESHOLD_REF * scale_y)

        # White detection using relaxed threshold approach
        # The game has green-tinted selection highlights that reduce R and B channels
        # while G stays high. Example: R=227, G=254, B=219 for green-tinted white
        #
        # Strategy: Detect bright pixels where at least one channel is very bright
        # and the average/luminance is reasonably high
        b, g, r = cv2.split(cv_image)

        # Calculate luminance (approximate) - using float to avoid overflow
        luminance = (0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32))

        # Detect white/bright digits:
        # 1. Pure white: all channels > 220
        # 2. Green-tinted: G > 240 AND luminance > 190 (catches selection highlights)
        # 3. High luminance with any bright channel: luminance > 200 AND max channel > 230
        # 4. Very high single channel: any channel > 248 AND luminance > 180
        pure_white = ((b > 220) & (g > 220) & (r > 220))
        green_tinted = ((g > 240) & (luminance > 190))
        max_channel = np.maximum(np.maximum(r, g), b)
        high_lum_bright = ((luminance > 200) & (max_channel > 230))
        very_bright_channel = ((max_channel > 248) & (luminance > 180))

        binary = (pure_white | green_tinted | high_lum_bright | very_bright_channel).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and location
        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Check size constraints (using scaled values)
            if not (min_cell_area < area < max_cell_area):
                continue
            if not (min_cell_width < w < max_cell_width):
                continue
            if not (min_cell_height < h < max_cell_height):
                continue

            # Check grid region bounds (using scaled values)
            if not (grid_x_min < x < grid_x_max):
                continue
            if not (grid_y_min < y < grid_y_max):
                continue

            # Extract cell image
            roi = binary[y:y+h, x:x+w]
            cells.append((x, y, w, h, roi))

        if len(cells) < EXPECTED_CELLS * 0.3:  # Lower threshold - may have many empty cells
            return None, [], None

        # Sort cells into rows by y-coordinate
        cells.sort(key=lambda c: c[1])
        raw_rows: List[List[Tuple]] = []
        current_row = [cells[0]]

        for cell in cells[1:]:
            if abs(cell[1] - current_row[0][1]) < row_grouping_threshold:
                current_row.append(cell)
            else:
                current_row.sort(key=lambda c: c[0])
                raw_rows.append(current_row)
                current_row = [cell]

        current_row.sort(key=lambda c: c[0])
        raw_rows.append(current_row)

        # Validate grid structure
        if len(raw_rows) < EXPECTED_ROWS * 0.5:
            return None, [], None

        # Calculate cell statistics
        all_widths = [c[2] for row in raw_rows for c in row]
        all_heights = [c[3] for row in raw_rows for c in row]
        cell_size = int(max(np.median(all_widths), np.median(all_heights)))

        # Calculate stride (distance between adjacent cell centers)
        x_diffs = []
        for row in raw_rows:
            row_sorted = sorted(row, key=lambda c: c[0])
            for i in range(len(row_sorted) - 1):
                diff = row_sorted[i + 1][0] - row_sorted[i][0]
                # Only consider adjacent cells (within 2x expected stride)
                if 0 < diff < cell_size * 3:
                    x_diffs.append(diff)

        cell_stride = int(np.median(x_diffs)) if x_diffs else cell_size + 4

        # Calculate row Y positions (median Y for each row)
        row_y_positions = []
        for row in raw_rows:
            median_y = int(np.median([c[1] + c[3] // 2 for c in row]))
            row_y_positions.append(median_y)

        # Determine grid origin (leftmost x position across all cells)
        # Use the minimum detected x as the origin for column 0
        all_x_positions = [c[0] for row in raw_rows for c in row]
        grid_x_origin = min(all_x_positions)

        # Build fixed-column grid
        cell_positions: List[List[Tuple[int, int]]] = []
        cell_images: List[List[Optional[np.ndarray]]] = []

        for row_idx, row in enumerate(raw_rows):
            # Create a lookup for detected cells by their column index
            cell_by_col: dict[int, Tuple] = {}
            for cell in row:
                x, y, w, h, roi = cell
                cell_center_x = x + w // 2
                # Calculate which column this cell belongs to
                col_idx = round((x - grid_x_origin) / cell_stride)
                if 0 <= col_idx < EXPECTED_COLS:
                    cell_by_col[col_idx] = cell

            # Build fixed-width row with None for empty cells
            row_positions = []
            row_images = []
            row_y = row_y_positions[row_idx] if row_idx < len(row_y_positions) else row_y_positions[-1]

            for col_idx in range(EXPECTED_COLS):
                expected_x = grid_x_origin + col_idx * cell_stride + cell_size // 2

                if col_idx in cell_by_col:
                    x, y, w, h, roi = cell_by_col[col_idx]
                    row_positions.append((x + w // 2, y + h // 2))
                    row_images.append(roi)
                else:
                    # Empty cell - use calculated position, None for image
                    row_positions.append((expected_x, row_y))
                    row_images.append(None)

            cell_positions.append(row_positions)
            cell_images.append(row_images)

        # Calculate grid bounds
        min_x = grid_x_origin
        min_y = row_y_positions[0] - cell_size // 2 if row_y_positions else grid_y_min
        max_x = grid_x_origin + (EXPECTED_COLS - 1) * cell_stride + cell_size
        max_y = row_y_positions[-1] + cell_size // 2 if row_y_positions else grid_y_max

        grid_info = GridInfo(
            bounds=(min_x, min_y, max_x - min_x, max_y - min_y),
            rows=len(raw_rows),
            cols=EXPECTED_COLS,
            cell_size=cell_size,
            cell_gap=max(0, cell_stride - cell_size),
            cell_positions=cell_positions,
            origin=(grid_x_origin, min_y),
            cell_stride=cell_stride
        )

        return grid_info, cell_images, cv_image

    def _extract_board(
        self,
        cell_images: List[List[Optional[np.ndarray]]],
        grid_info: GridInfo,
        start_time: float,
        original_image: Optional[np.ndarray] = None
    ) -> OCRResult:
        """Extract board values using template matching.

        Empty cells (None images) are validated against the original image
        to ensure the whole cell region matches background (truly empty)
        vs having content that failed threshold detection.
        """
        board: List[List[int]] = []
        cell_results: List[CellResult] = []
        total_confidence = 0.0
        uncertain_count = 0

        for row_idx, (row_images, row_positions) in enumerate(
            zip(cell_images, grid_info.cell_positions)
        ):
            board_row: List[int] = []

            for col_idx, (cell_img, (cx, cy)) in enumerate(zip(row_images, row_positions)):
                if cell_img is None:
                    # No white contour found - validate if cell region is truly empty
                    # by checking if the whole cell has uniform background color
                    digit = 0
                    confidence = 1.0
                    scores = []

                    if original_image is not None:
                        # Extract cell region from original image
                        half_size = grid_info.cell_size // 2 + CELL_REGION_PADDING
                        y1 = max(0, cy - half_size)
                        y2 = min(original_image.shape[0], cy + half_size)
                        x1 = max(0, cx - half_size)
                        x2 = min(original_image.shape[1], cx + half_size)

                        if x2 > x1 and y2 > y1:
                            cell_region = original_image[y1:y2, x1:x2]

                            # Calculate variance of the cell region
                            # Low variance = uniform background = truly empty
                            # High variance = content present that failed detection
                            variance = np.var(cell_region)

                            if variance > EMPTY_CELL_VARIANCE_THRESHOLD:
                                # High variance indicates content present
                                # This cell likely has a digit that failed threshold detection
                                confidence = 0.3  # Low confidence - flag as uncertain
                                uncertain_count += 1
                else:
                    digit, confidence, scores = self._templates.match(cell_img)
                    if digit is None:
                        digit = 0  # Unrecognized becomes 0
                        uncertain_count += 1

                board_row.append(digit)
                total_confidence += confidence

                if digit != 0 and confidence < CONFIDENCE_THRESHOLD:
                    uncertain_count += 1

                cell_results.append(CellResult(
                    row=row_idx,
                    col=col_idx,
                    value=digit if digit != 0 else None,  # Keep None for display purposes
                    confidence=confidence,
                    position=(cx, cy),
                    raw_scores=scores
                ))

            board.append(board_row)

        total_cells = len(cell_results)
        avg_confidence = total_confidence / total_cells if total_cells > 0 else 0.0

        processing_time = (time.perf_counter() - start_time) * 1000

        return OCRResult(
            board=board,
            confidence=avg_confidence,
            cell_results=cell_results,
            grid_info=grid_info,
            uncertain_count=uncertain_count,
            total_cells=total_cells,
            processing_time_ms=processing_time
        )

    @property
    def templates(self) -> DigitTemplates:
        """Access to digit templates for external tools."""
        return self._templates


def _preprocess_cell(cell_image: np.ndarray) -> np.ndarray:
    """Apply preprocessing pipeline to a cell image for digit recognition.

    Uses bright white RGB detection (all channels > 232) to match
    the main OCR engine's detection method.
    """
    if len(cell_image.shape) == 3:
        # Use same bright white RGB detection as main engine
        b, g, r = cv2.split(cell_image)
        binary = ((b > 232) & (g > 232) & (r > 232)).astype(np.uint8) * 255
        return binary
    else:
        # Grayscale fallback
        _, thresh = cv2.threshold(cell_image, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
        return thresh


# Module-level helper for backward compatibility
def detect_grid(image: Image.Image) -> Optional[GridInfo]:
    """
    Standalone grid detection function for backward compatibility.

    Args:
        image: PIL Image of the game window

    Returns:
        GridInfo if grid detected, None otherwise
    """
    engine = TemplateOCREngine()
    grid_info, _, _ = engine._detect_grid(image)
    return grid_info


# Expose preprocessing function
preprocess_cell = _preprocess_cell
