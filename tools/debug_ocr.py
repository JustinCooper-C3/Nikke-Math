"""
Diagnostic script to analyze OCR detection on debug images.
Outputs variance values for cells marked as empty to understand false zeros.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Constants from template_engine
WHITE_THRESHOLD = 232
EMPTY_CELL_VARIANCE_THRESHOLD = 500
CELL_REGION_PADDING = 5
GRID_X_MIN = 650
GRID_X_MAX = 1150
GRID_Y_MIN = 230
GRID_Y_MAX = 1000
MIN_CELL_AREA = 15
MAX_CELL_AREA = 800
ROW_GROUPING_THRESHOLD = 15
EXPECTED_COLS = 10


def analyze_image(image_path: str):
    """Analyze an image and report variance for all cell positions."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path}")
    print(f"{'='*60}")

    img = Image.open(image_path)
    cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # White detection using relaxed threshold (matches updated template_engine)
    b, g, r = cv2.split(cv_image)

    # Calculate luminance (approximate) - using float to avoid overflow
    luminance = (0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32))

    # Detect white/bright digits:
    pure_white = ((b > 220) & (g > 220) & (r > 220))
    green_tinted = ((g > 240) & (luminance > 190))
    max_channel = np.maximum(np.maximum(r, g), b)
    high_lum_bright = ((luminance > 200) & (max_channel > 230))
    very_bright_channel = ((max_channel > 248) & (luminance > 180))

    binary = (pure_white | green_tinted | high_lum_bright | very_bright_channel).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and collect cells
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if not (MIN_CELL_AREA < area < MAX_CELL_AREA):
            continue
        if not (5 < w < 30):
            continue
        if not (8 < h < 35):
            continue
        if not (GRID_X_MIN < x < GRID_X_MAX):
            continue
        if not (GRID_Y_MIN < y < GRID_Y_MAX):
            continue

        cells.append((x, y, w, h))

    print(f"Found {len(cells)} white digit contours")

    # Sort into rows
    cells.sort(key=lambda c: c[1])
    rows = []
    current_row = [cells[0]] if cells else []

    for cell in cells[1:]:
        if abs(cell[1] - current_row[0][1]) < ROW_GROUPING_THRESHOLD:
            current_row.append(cell)
        else:
            current_row.sort(key=lambda c: c[0])
            rows.append(current_row)
            current_row = [cell]

    if current_row:
        current_row.sort(key=lambda c: c[0])
        rows.append(current_row)

    print(f"Organized into {len(rows)} rows")

    # Calculate grid origin and stride
    all_x = [c[0] for row in rows for c in row]
    grid_x_origin = min(all_x)

    x_diffs = []
    for row in rows:
        row_sorted = sorted(row, key=lambda c: c[0])
        for i in range(len(row_sorted) - 1):
            diff = row_sorted[i + 1][0] - row_sorted[i][0]
            if 0 < diff < 100:
                x_diffs.append(diff)

    cell_stride = int(np.median(x_diffs)) if x_diffs else 40
    cell_size = 17  # From logs

    print(f"Grid origin X: {grid_x_origin}, stride: {cell_stride}")

    # Analyze each row
    print(f"\n--- Cell Analysis ---")
    print(f"{'Row':>3} {'Col':>3} {'Status':>10} {'Variance':>10} {'RGB_Max':>10}")
    print("-" * 50)

    for row_idx, row in enumerate(rows[:4]):  # First 4 rows
        row_y = int(np.median([c[1] + c[3] // 2 for c in row]))

        # Build detected column map
        detected_cols = {}
        for cell in row:
            x, y, w, h = cell
            col_idx = round((x - grid_x_origin) / cell_stride)
            if 0 <= col_idx < EXPECTED_COLS:
                detected_cols[col_idx] = cell

        # Check each column
        for col_idx in range(EXPECTED_COLS):
            expected_x = grid_x_origin + col_idx * cell_stride + cell_size // 2

            if col_idx in detected_cols:
                status = "DETECTED"
                variance = 0
                rgb_max = 255
            else:
                status = "EMPTY"
                # Calculate variance for this position
                half_size = cell_size // 2 + CELL_REGION_PADDING
                y1 = max(0, row_y - half_size)
                y2 = min(cv_image.shape[0], row_y + half_size)
                x1 = max(0, expected_x - half_size)
                x2 = min(cv_image.shape[1], expected_x + half_size)

                cell_region = cv_image[y1:y2, x1:x2]
                variance = np.var(cell_region)

                # Also check max RGB values in region
                rgb_max = np.max(cell_region)

            # Flag suspicious empties
            flag = ""
            if status == "EMPTY" and variance > EMPTY_CELL_VARIANCE_THRESHOLD:
                flag = " <-- HIGH VARIANCE (likely false zero!)"

            print(f"{row_idx:>3} {col_idx:>3} {status:>10} {variance:>10.1f} {rgb_max:>10}{flag}")

    # Also show what RGB values we're seeing for non-detected regions
    print(f"\n--- Sample pixel values in suspicious regions ---")
    for row_idx, row in enumerate(rows[:4]):
        row_y = int(np.median([c[1] + c[3] // 2 for c in row]))
        detected_cols = {}
        for cell in row:
            x, y, w, h = cell
            col_idx = round((x - grid_x_origin) / cell_stride)
            if 0 <= col_idx < EXPECTED_COLS:
                detected_cols[col_idx] = cell

        for col_idx in range(EXPECTED_COLS):
            if col_idx not in detected_cols:
                expected_x = grid_x_origin + col_idx * cell_stride + cell_size // 2
                half_size = cell_size // 2 + CELL_REGION_PADDING
                y1 = max(0, row_y - half_size)
                y2 = min(cv_image.shape[0], row_y + half_size)
                x1 = max(0, expected_x - half_size)
                x2 = min(cv_image.shape[1], expected_x + half_size)

                cell_region = cv_image[y1:y2, x1:x2]
                variance = np.var(cell_region)

                if variance > 200:  # Suspicious
                    # Get RGB stats
                    b_ch = cell_region[:,:,0]
                    g_ch = cell_region[:,:,1]
                    r_ch = cell_region[:,:,2]
                    print(f"Row {row_idx}, Col {col_idx}: variance={variance:.0f}")
                    print(f"  R: min={r_ch.min()}, max={r_ch.max()}, mean={r_ch.mean():.0f}")
                    print(f"  G: min={g_ch.min()}, max={g_ch.max()}, mean={g_ch.mean():.0f}")
                    print(f"  B: min={b_ch.min()}, max={b_ch.max()}, mean={b_ch.mean():.0f}")

                    # Check how many pixels pass threshold
                    white_mask = (r_ch > 232) & (g_ch > 232) & (b_ch > 232)
                    white_count = np.sum(white_mask)
                    total = white_mask.size
                    print(f"  White pixels (>232): {white_count}/{total} ({100*white_count/total:.1f}%)")


if __name__ == "__main__":
    # Analyze recent clean debug images
    debug_dir = Path("debug")
    clean_images = sorted(debug_dir.glob("debug_*_clean.png"))[-3:]

    if not clean_images:
        # Fall back to older naming
        clean_images = sorted(debug_dir.glob("overlay_debug_*.png"))[-3:]

    if not clean_images:
        print("No debug images found!")
    else:
        for img_path in clean_images:
            analyze_image(str(img_path))
