#!/usr/bin/env python3
"""
Template extraction tool for OCR calibration.

Extracts digit templates from a debug image where all digits 1-9 are visible.
Saves templates to assets/templates/ for improved OCR accuracy.

Usage:
    python extract_templates.py [image_path]

The script will:
1. Detect the grid and all cells
2. Display each cell and ask you to label it (1-9 or skip)
3. Average multiple samples of each digit
4. Save templates to assets/templates/

Examples:
    python extract_templates.py debug/debug_20251211.png
"""

import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr import detect_grid, preprocess_cell


TEMPLATE_DIR = Path("./assets/templates")


def extract_cells(image_path: str) -> list:
    """
    Extract all preprocessed cells from an image.

    Returns list of (cell_image, position) tuples.
    """
    image = Image.open(image_path)
    grid_info = detect_grid(image)

    if not grid_info:
        print("ERROR: Could not detect grid")
        return []

    print(f"Grid detected: {grid_info.rows}x{grid_info.cols}, cell size: {grid_info.cell_size}px")

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cells = []
    half_size = grid_info.cell_size // 2
    margin = int(half_size * 0.2)

    for row_idx, row in enumerate(grid_info.cell_positions):
        for col_idx, (cx, cy) in enumerate(row):
            if cx < 0 or cy < 0:
                continue

            x1 = max(0, cx - half_size + margin)
            y1 = max(0, cy - half_size + margin)
            x2 = min(cv_image.shape[1], cx + half_size - margin)
            y2 = min(cv_image.shape[0], cy + half_size - margin)

            cell_img = cv_image[y1:y2, x1:x2]
            if cell_img.size == 0:
                continue

            preprocessed = preprocess_cell(cell_img)
            cells.append((preprocessed, cell_img, (row_idx, col_idx)))

    return cells, grid_info.cell_size


def auto_label_from_board(image_path: str) -> dict:
    """
    Attempt to auto-label cells based on known board state.

    This requires manual verification but speeds up the process.
    """
    # For now, return empty - manual labeling required
    return {}


def interactive_label(cells: list) -> dict:
    """
    Interactively label cells by showing them to the user.

    Returns dict mapping digit -> list of cell images.
    """
    digit_samples = defaultdict(list)

    print("\n" + "="*60)
    print("Interactive Template Labeling")
    print("="*60)
    print("For each cell shown, enter the digit (1-9), 's' to skip, or 'q' to quit.")
    print("The more samples per digit, the better the templates.")
    print()

    cv2.namedWindow("Cell", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cell", 200, 200)

    for i, (preprocessed, original, pos) in enumerate(cells):
        # Show both original and preprocessed
        display = np.hstack([
            cv2.resize(original, (100, 100)),
            cv2.resize(cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR), (100, 100))
        ])

        cv2.imshow("Cell", display)

        print(f"Cell {i+1}/{len(cells)} at {pos} - Enter digit (1-9), 's' to skip, 'q' to quit: ", end="", flush=True)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("quit")
                cv2.destroyAllWindows()
                return digit_samples
            elif key == ord('s'):
                print("skipped")
                break
            elif ord('1') <= key <= ord('9'):
                digit = key - ord('0')
                digit_samples[digit].append(preprocessed)
                print(f"{digit} (total samples for {digit}: {len(digit_samples[digit])})")
                break
            else:
                print(f"\n  Invalid key. Enter 1-9, 's', or 'q': ", end="", flush=True)

    cv2.destroyAllWindows()
    return digit_samples


def create_templates(digit_samples: dict, cell_size: int) -> dict:
    """
    Create averaged templates from samples.

    Returns dict mapping digit -> template image.
    """
    templates = {}

    for digit, samples in digit_samples.items():
        if not samples:
            continue

        # Resize all samples to same size
        resized = [cv2.resize(s, (cell_size, cell_size)) for s in samples]

        # Average the samples
        stacked = np.stack(resized, axis=0).astype(np.float32)
        averaged = np.mean(stacked, axis=0).astype(np.uint8)

        # Threshold to clean up
        _, template = cv2.threshold(averaged, 127, 255, cv2.THRESH_BINARY)

        templates[digit] = template
        print(f"Digit {digit}: {len(samples)} samples averaged")

    return templates


def save_templates(templates: dict):
    """Save templates to assets/templates/."""
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    for digit, template in templates.items():
        path = TEMPLATE_DIR / f"{digit}.png"
        cv2.imwrite(str(path), template)
        print(f"Saved: {path}")


def quick_extract(image_path: str):
    """
    Quick extraction mode - extract samples of each unique digit pattern.

    Uses clustering to find unique patterns, then asks for labels.
    """
    cells, cell_size = extract_cells(image_path)

    if not cells:
        return

    print(f"\nExtracted {len(cells)} cells")

    # Cluster similar cells
    from sklearn.cluster import KMeans

    # Flatten cells for clustering
    flat_cells = []
    for preprocessed, original, pos in cells:
        resized = cv2.resize(preprocessed, (20, 20))
        flat_cells.append(resized.flatten())

    X = np.array(flat_cells)

    # Cluster into ~15 groups (some digits may have multiple variants)
    n_clusters = min(15, len(cells))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Get representative sample from each cluster
    representatives = {}
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        # Pick the one closest to centroid
        cluster_cells = X[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_cells - centroid, axis=1)
        best_idx = cluster_indices[np.argmin(distances)]
        representatives[cluster_id] = cells[best_idx]

    print(f"Found {n_clusters} unique patterns")

    # Label representatives
    digit_samples = defaultdict(list)

    cv2.namedWindow("Pattern", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pattern", 200, 200)

    for cluster_id, (preprocessed, original, pos) in representatives.items():
        # Count samples in this cluster
        count = np.sum(labels == cluster_id)

        display = np.hstack([
            cv2.resize(original, (100, 100)),
            cv2.resize(cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR), (100, 100))
        ])

        cv2.imshow("Pattern", display)

        print(f"Pattern {cluster_id+1}/{n_clusters} ({count} samples) - Digit (1-9), 's' skip, 'q' quit: ", end="", flush=True)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("quit")
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                print("skipped")
                break
            elif ord('1') <= key <= ord('9'):
                digit = key - ord('0')
                # Add all samples from this cluster
                for idx in np.where(labels == cluster_id)[0]:
                    digit_samples[digit].append(cells[idx][0])
                print(f"{digit} ({count} samples added)")
                break
        else:
            continue
        break
    else:
        cv2.destroyAllWindows()

    # Create and save templates
    if digit_samples:
        templates = create_templates(digit_samples, cell_size)
        save_templates(templates)
        print(f"\nTemplates saved to {TEMPLATE_DIR}/")
    else:
        print("\nNo samples collected")


def main():
    if len(sys.argv) < 2:
        # Find most recent debug image
        debug_dir = Path("./debug")
        debug_images = list(debug_dir.glob("debug_*.png"))
        if debug_images:
            debug_images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            image_path = str(debug_images[0])
        else:
            print("Usage: python extract_templates.py <image_path>")
            print("\nNo debug images found. Run the solver first to capture images.")
            return 1
    else:
        image_path = sys.argv[1]

    print(f"Using image: {image_path}")

    try:
        quick_extract(image_path)
    except ImportError:
        print("sklearn not installed, using manual labeling mode")
        cells, cell_size = extract_cells(image_path)
        if cells:
            digit_samples = interactive_label(cells)
            if digit_samples:
                templates = create_templates(dict(digit_samples), cell_size)
                save_templates(templates)

    return 0


if __name__ == "__main__":
    sys.exit(main())
