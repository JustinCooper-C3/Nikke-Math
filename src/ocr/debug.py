"""
OCR Debug Utilities

Functions for saving annotated debug images and managing debug output.
"""

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .result import GridInfo, OCRResult


# Debug settings
DEBUG_DIR = Path("./debug")
MAX_DEBUG_IMAGES = 10

# Confidence thresholds for coloring
HIGH_CONFIDENCE = 0.95
MEDIUM_CONFIDENCE = 0.80


def save_debug_image(
    image: Image.Image,
    grid_info: Optional[GridInfo],
    result: Optional[OCRResult],
    path: str
) -> None:
    """
    Save an annotated debug image showing grid detection and OCR results.

    Annotations include:
    - Grid boundary box
    - Cell center markers
    - Recognized digits with confidence colors
    - Uncertain cells highlighted in red

    Args:
        image: Original PIL Image
        grid_info: Grid detection result (can be None)
        result: OCR result (can be None)
        path: Output file path
    """
    # Ensure debug directory exists
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Create a copy to draw on
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        small_font = ImageFont.truetype("arial.ttf", 10)
    except:
        font = ImageFont.load_default()
        small_font = font

    if grid_info:
        # Draw grid bounds
        bx, by, bw, bh = grid_info.bounds
        draw.rectangle([bx, by, bx + bw, by + bh], outline="blue", width=2)

        # Draw info text
        info_text = f"Grid: {grid_info.rows}x{grid_info.cols}, Cell: {grid_info.cell_size}px"
        draw.text((10, 10), info_text, fill="blue", font=font)

    if result and result.cell_results:
        for cell in result.cell_results:
            cx, cy = cell.position

            if cx < 0 or cy < 0:
                continue

            # Choose color based on confidence
            if cell.confidence >= HIGH_CONFIDENCE:
                color = "green"
            elif cell.confidence >= MEDIUM_CONFIDENCE:
                color = "yellow"
            else:
                color = "red"

            # Draw recognized digit (no dot markers)
            if cell.value is not None:
                text = str(cell.value)
                draw.text((cx + 5, cy - 5), text, fill=color, font=small_font)
            else:
                draw.text((cx + 5, cy - 5), "?", fill="red", font=small_font)

        # Draw summary
        summary = f"Cells: {result.total_cells}, Uncertain: {result.uncertain_count}, " \
                  f"Confidence: {result.confidence * 100:.1f}%, Time: {result.processing_time_ms:.1f}ms"
        draw.text((10, 30), summary, fill="blue", font=font)

    # Save image
    debug_img.save(path, "PNG")

    # Cleanup old debug images
    _cleanup_debug_images()


def _cleanup_debug_images() -> None:
    """Remove old debug images, keeping only the most recent MAX_DEBUG_IMAGES."""
    if not DEBUG_DIR.exists():
        return

    # Get all debug images sorted by modification time
    debug_files = sorted(
        DEBUG_DIR.glob("debug_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    # Remove old files
    for old_file in debug_files[MAX_DEBUG_IMAGES:]:
        try:
            old_file.unlink()
        except OSError:
            pass


def get_confidence_color(confidence: float) -> str:
    """
    Get color code for confidence level.

    Args:
        confidence: Confidence value 0.0-1.0

    Returns:
        Hex color code string
    """
    if confidence >= HIGH_CONFIDENCE:
        return "#4CAF50"  # Green
    elif confidence >= MEDIUM_CONFIDENCE:
        return "#FFC107"  # Yellow
    else:
        return "#d32f2f"  # Red
