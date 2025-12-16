# PythonOverlayLib API Reference

API documentation for PythonOverlayLib - a PyQt5-based library for creating transparent, always-on-top overlay windows.

**Repository:** [LUXTACO/PythonOverlayLib](https://github.com/LUXTACO/PythonOverlayLib)
**License:** Apache License 2.0
**Status:** Archived (read-only) as of April 2025, but functional

---

## Installation

```bash
pip install PythonOverlayLib
```

**Dependency:** Requires PyQt5 (installed automatically)

---

## Core Classes

### Overlay

Main class for creating the transparent overlay window.

```python
import overlay_lib

overlay = overlay_lib.Overlay(
    drawlistCallback=callback_function,  # Function that returns list of shapes
    refreshTimeout=16                     # Update interval in milliseconds (16ms = ~60fps)
)
overlay.spawn()  # Start the overlay
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `drawlistCallback` | `Callable` | Function returning a list of shape objects to draw |
| `refreshTimeout` | `int` | Milliseconds between callback executions |

---

## Utility Classes

### Vector2D

Represents a 2D coordinate position.

```python
from overlay_lib import Vector2D

position = Vector2D(x, y)
# Example: Vector2D(960, 540) - center of 1920x1080 screen
```

### Size2D

Represents width and height dimensions.

```python
from overlay_lib import Size2D

size = Size2D(width, height)
# Example: Size2D(100, 50) - 100px wide, 50px tall
```

### RgbaColor

Defines a color with red, green, blue, and alpha (transparency) channels.

```python
from overlay_lib import RgbaColor

color = RgbaColor(red, green, blue, alpha)
# Values: 0-255 for each channel
# Example: RgbaColor(255, 0, 0, 128) - semi-transparent red
```

### RgbaGradient

Defines a gradient color (for filled shapes).

```python
from overlay_lib import RgbaGradient

gradient = RgbaGradient(color1, color2)
```

---

## Shape Classes

### SkDrawCircle

Draws a stroked (outline) circle.

```python
from overlay_lib import SkDrawCircle, Vector2D, RgbaColor

circle = SkDrawCircle(
    position,   # Vector2D - center point
    radius,     # int - radius in pixels
    color,      # RgbaColor - stroke color
    thickness   # int - line thickness in pixels
)
```

### SkDrawRect

Draws a stroked (outline) rectangle.

```python
from overlay_lib import SkDrawRect, Vector2D, Size2D, RgbaColor

rect = SkDrawRect(
    position,   # Vector2D - top-left corner
    size,       # Size2D - width and height
    color,      # RgbaColor - stroke color
    thickness   # int - line thickness in pixels
)
```

### FlDrawCircle

Draws a filled circle.

```python
from overlay_lib import FlDrawCircle, Vector2D, RgbaColor

filled_circle = FlDrawCircle(
    position,   # Vector2D - center point
    radius,     # int - radius in pixels
    color       # RgbaColor or RgbaGradient - fill color
)
```

### FlDrawRect

Draws a filled rectangle.

```python
from overlay_lib import FlDrawRect, Vector2D, Size2D, RgbaColor

filled_rect = FlDrawRect(
    position,   # Vector2D - top-left corner
    size,       # Size2D - width and height
    color       # RgbaColor or RgbaGradient - fill color
)
```

### DrawLine

Draws a line between two points.

```python
from overlay_lib import DrawLine, Vector2D, RgbaColor

line = DrawLine(
    start,      # Vector2D - start point
    end,        # Vector2D - end point
    color,      # RgbaColor - line color
    thickness   # int - line thickness in pixels
)
```

### DrawText

Draws text at a position.

```python
from overlay_lib import DrawText, Vector2D, RgbaColor

text = DrawText(
    position,   # Vector2D - text position
    text,       # str - text content
    color,      # RgbaColor - text color
    font_size   # int - font size in points
)
```

### DrawImage

Draws an image at a position.

```python
from overlay_lib import DrawImage, Vector2D, Size2D

image = DrawImage(
    position,   # Vector2D - top-left corner
    size,       # Size2D - display size
    image_path  # str - path to image file
)
```

---

## Project Usage Example

Example for highlighting grid cells in the Nikke Math game:

```python
import overlay_lib
from overlay_lib import (
    Overlay, Vector2D, Size2D, RgbaColor,
    SkDrawRect, FlDrawRect, DrawText
)

# Grid configuration
GRID_ORIGIN = Vector2D(100, 200)  # Top-left of game grid
CELL_SIZE = Size2D(50, 50)        # Each cell is 50x50 pixels

# Colors
HIGHLIGHT_COLOR = RgbaColor(0, 255, 0, 180)      # Green, semi-transparent
SUGGESTION_COLOR = RgbaColor(255, 255, 0, 150)   # Yellow, semi-transparent
TEXT_COLOR = RgbaColor(255, 255, 255, 255)       # White, opaque

# Track which cells to highlight
highlighted_cells = []  # List of (row, col) tuples
suggested_sum = 0

def get_cell_position(row, col):
    """Convert grid coordinates to screen position."""
    x = GRID_ORIGIN.x + (col * CELL_SIZE.width)
    y = GRID_ORIGIN.y + (row * CELL_SIZE.height)
    return Vector2D(x, y)

def draw_callback():
    """Called every frame to return shapes to draw."""
    shapes = []

    # Draw highlight rectangles for selected cells
    for row, col in highlighted_cells:
        pos = get_cell_position(row, col)
        rect = FlDrawRect(pos, CELL_SIZE, HIGHLIGHT_COLOR)
        shapes.append(rect)

    # Draw border around selection area
    if highlighted_cells:
        # Calculate bounding box
        min_row = min(c[0] for c in highlighted_cells)
        max_row = max(c[0] for c in highlighted_cells)
        min_col = min(c[1] for c in highlighted_cells)
        max_col = max(c[1] for c in highlighted_cells)

        top_left = get_cell_position(min_row, min_col)
        selection_size = Size2D(
            (max_col - min_col + 1) * CELL_SIZE.width,
            (max_row - min_row + 1) * CELL_SIZE.height
        )
        border = SkDrawRect(top_left, selection_size, SUGGESTION_COLOR, 3)
        shapes.append(border)

        # Show sum text
        text_pos = Vector2D(top_left.x, top_left.y - 25)
        sum_text = DrawText(text_pos, f"Sum: {suggested_sum}", TEXT_COLOR, 16)
        shapes.append(sum_text)

    return shapes

# Create and start overlay
overlay = Overlay(
    drawlistCallback=draw_callback,
    refreshTimeout=16  # ~60 FPS
)

# Start overlay (blocks main thread)
overlay.spawn()
```

---

## Notes

- The overlay is transparent and click-through by default
- `spawn()` blocks the main thread - run game logic in separate thread if needed
- Refresh timeout of 16ms provides ~60 FPS updates
- All coordinates are screen-absolute positions
- The library uses PyQt5's event loop internally
