# OCR Engine Implementation Guide

This guide explains how to add new OCR engines to the Nikke Math Solver.

## Architecture Overview

The `src/ocr/` module provides a pluggable OCR architecture:

```
src/ocr/
    __init__.py          # Public exports
    base.py              # OCREngine abstract base class
    result.py            # Result dataclasses
    template_engine.py   # Default template matching implementation
    factory.py           # Engine factory and registration
    debug.py             # Debug utilities
```

## Quick Start

### Using the Default Engine

```python
from src.ocr import create_engine

engine = create_engine("template")
result = engine.process(image)
board = result.board  # List[List[Optional[int]]]
```

### Creating a Custom Engine

```python
from src.ocr import OCREngine, OCRResult, GridInfo, CellResult, register_engine
from PIL import Image

class MyOCREngine(OCREngine):
    @property
    def name(self) -> str:
        return "my_engine"

    def process(self, image: Image.Image) -> OCRResult:
        # Your implementation here
        ...

# Register and use
register_engine("my_engine", MyOCREngine)
engine = create_engine("my_engine")
```

## Required Interface

All OCR engines must inherit from `OCREngine` and implement:

### 1. `name` Property (Required)

```python
@property
def name(self) -> str:
    """Return unique engine identifier."""
    return "my_engine"
```

### 2. `process()` Method (Required)

```python
def process(self, image: Image.Image) -> OCRResult:
    """
    Process an image and extract the game board.

    Args:
        image: PIL Image of the game window

    Returns:
        OCRResult with board values and metadata
    """
```

### 3. `configure()` Method (Optional)

```python
def configure(self, **kwargs) -> None:
    """
    Configure engine parameters at runtime.

    Override to support dynamic configuration.
    Default implementation does nothing.
    """
    pass
```

## Result Data Structures

### OCRResult

The main return type from `process()`:

```python
@dataclass
class OCRResult:
    board: List[List[Optional[int]]]  # 2D array of digits (1-9) or None
    confidence: float                  # Average confidence (0.0-1.0)
    cell_results: List[CellResult]     # Per-cell details
    grid_info: Optional[GridInfo]      # Grid layout info (can be None)
    uncertain_count: int               # Cells below confidence threshold
    total_cells: int                   # Total cells detected
    processing_time_ms: float          # Processing duration
```

### GridInfo

Grid layout information for overlay rendering:

```python
@dataclass
class GridInfo:
    bounds: Tuple[int, int, int, int]  # (x, y, width, height)
    rows: int
    cols: int
    cell_size: int                     # Cell diameter in pixels
    cell_gap: int                      # Gap between cells
    cell_positions: List[List[Tuple[int, int]]]  # [row][col] = (x, y)
    origin: Optional[Tuple[int, int]]  # Top-left reference point
    cell_stride: int                   # Distance between cell centers
```

### CellResult

Per-cell analysis details:

```python
@dataclass
class CellResult:
    row: int
    col: int
    value: Optional[int]       # 1-9 or None
    confidence: float          # 0.0-1.0
    position: Tuple[int, int]  # (x, y) center
    raw_scores: List[float]    # Match scores for digits 1-9
```

## Registration and Usage

### Registering Your Engine

```python
from src.ocr import register_engine

register_engine("my_engine", MyOCREngine)
```

### Creating Engine Instances

```python
from src.ocr import create_engine, available_engines

# List available engines
print(available_engines())  # ["template", "my_engine"]

# Create with default settings
engine = create_engine("my_engine")

# Create with configuration
engine = create_engine("my_engine", some_param="value")
```

### Runtime Configuration

```python
engine = create_engine("my_engine")
engine.configure(threshold=0.8, debug=True)
```

## Error Handling Guidelines

### Grid Detection Failure

If grid cannot be detected, return a valid `OCRResult` with empty data:

```python
if grid_info is None:
    return OCRResult(
        board=[],
        confidence=0.0,
        cell_results=[],
        grid_info=None,
        uncertain_count=0,
        total_cells=0,
        processing_time_ms=elapsed
    )
```

### Individual Cell Failures

For cells that cannot be recognized:
- Set `value=None` in the board
- Set `confidence=0.0` in CellResult
- Increment `uncertain_count`

```python
if not recognized:
    board_row.append(None)
    cell_results.append(CellResult(
        row=r, col=c,
        value=None,
        confidence=0.0,
        position=(cx, cy),
        raw_scores=[]
    ))
    uncertain_count += 1
```

## Testing Your Engine

```python
from PIL import Image
from src.ocr import create_engine

def test_engine():
    engine = create_engine("my_engine")

    # Load test image
    image = Image.open("debug/debug_sample.png")

    # Process
    result = engine.process(image)

    # Verify results
    assert result.board is not None
    assert result.total_cells > 0

    # Check recognition rate
    recognized = result.total_cells - result.uncertain_count
    print(f"Recognized: {recognized}/{result.total_cells}")
    print(f"Confidence: {result.confidence * 100:.1f}%")

    # Verify board structure
    for row in result.board:
        for cell in row:
            assert cell is None or (1 <= cell <= 9)

if __name__ == "__main__":
    test_engine()
```

## Best Practices

1. **Reuse grid detection**: The template engine's grid detection works well. Consider importing and reusing it.

2. **Handle edge cases**: Missing cells, partial grids, and failed recognition should all be handled gracefully.

3. **Provide confidence scores**: Even approximate confidence helps the solver make better decisions.

4. **Support configuration**: Allow users to tune parameters like thresholds without code changes.

5. **Track processing time**: Include in `processing_time_ms` for performance monitoring.

6. **Fill all CellResult fields**: Even if your engine doesn't use `raw_scores`, provide an empty list.

## Available Engines

| Engine | Type | Dependencies | Notes |
|--------|------|--------------|-------|
| `template` | Template Matching | OpenCV, NumPy | Default, no external OCR needed |

## API Reference

### Factory Functions

```python
create_engine(engine_type: str = "template", **config) -> OCREngine
register_engine(name: str, engine_class: type) -> None
available_engines() -> list[str]
```

### OCREngine Methods

```python
process(image: Image.Image) -> OCRResult  # Required
name: str  # Required property
configure(**kwargs) -> None  # Optional
```
