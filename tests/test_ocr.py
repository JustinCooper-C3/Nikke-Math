#!/usr/bin/env python3
"""
Test script for OCR accuracy validation.

Loads debug images and runs the OCR pipeline to measure accuracy.

Usage:
    python test_ocr.py [image_path]

Examples:
    python test_ocr.py                           # Test all debug images
    python test_ocr.py debug/debug_20251211.png  # Test specific image
"""

import sys
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr import (
    create_engine,
    detect_grid,
    save_debug_image,
    DEBUG_DIR,
)


def test_single_image(image_path: str) -> dict:
    """
    Test OCR on a single image.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print('='*60)

    # Load image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    # Create OCR engine and process
    engine = create_engine("template")
    result = engine.process(image)

    if result.grid_info is None:
        print("ERROR: Grid not detected!")
        return {"success": False, "error": "Grid not detected"}

    print(f"Grid: {result.grid_info.rows}x{result.grid_info.cols}")
    print(f"Cell size: {result.grid_info.cell_size}px")

    print(f"\nResults:")
    print(f"  Total cells: {result.total_cells}")
    print(f"  Uncertain: {result.uncertain_count}")
    print(f"  Confidence: {result.confidence * 100:.1f}%")
    print(f"  Processing time: {result.processing_time_ms:.1f}ms")

    # Count recognized digits
    recognized = sum(1 for row in result.board for cell in row if cell is not None)
    print(f"  Recognized: {recognized}/{result.total_cells}")

    # Show confidence distribution
    confidences = [c.confidence for c in result.cell_results]
    high = sum(1 for c in confidences if c >= 0.95)
    medium = sum(1 for c in confidences if 0.80 <= c < 0.95)
    low = sum(1 for c in confidences if c < 0.80)
    print(f"\nConfidence distribution:")
    print(f"  High (>=95%): {high}")
    print(f"  Medium (80-95%): {medium}")
    print(f"  Low (<80%): {low}")

    # Print board preview
    print(f"\nBoard preview (first 5 rows):")
    for i, row in enumerate(result.board[:5]):
        row_str = ' '.join(str(c) if c else '?' for c in row)
        print(f"  Row {i}: {row_str}")

    # Save new debug image
    output_path = DEBUG_DIR / f"test_{Path(image_path).stem}.png"
    save_debug_image(image, result.grid_info, result, str(output_path))
    print(f"\nDebug image saved: {output_path}")

    return {
        "success": True,
        "total_cells": result.total_cells,
        "recognized": recognized,
        "uncertain": result.uncertain_count,
        "confidence": result.confidence,
        "high_conf": high,
        "medium_conf": medium,
        "low_conf": low
    }


def main():
    """Main test function."""
    # Find debug images
    debug_images = list(DEBUG_DIR.glob("debug_*.png"))

    if len(sys.argv) > 1:
        # Test specific image
        test_single_image(sys.argv[1])
    elif debug_images:
        # Test most recent debug image
        debug_images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = debug_images[0]
        test_single_image(str(latest))
    else:
        print("No debug images found in ./debug/")
        print("Run the solver first to generate debug images, or specify an image path:")
        print("  python test_ocr.py path/to/image.png")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
