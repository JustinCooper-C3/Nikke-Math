# Nikke Math - Project Instructions

## Project Overview

A Python-based puzzle solver for a number grid game. The application captures a game window, reads the board state, and calculates optimal moves.

## Project Plan

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the full build plan, phases, and task tracking.

## Technology Stack

- **Language:** Python
- **Environment:** Virtual environment (venv)
- **Platform:** Windows
- **Overlay Framework:** PythonOverlayLib (PyQt5-based transparent click-through overlays) - See `docs/overlaylib-api.md`
- **Screen Capture:** Window capture/lock functionality
- **Image Processing:** OCR for reading grid numbers

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Game Rules

1. **Grid Structure:** The game board is a grid where each cell contains a number 1-9
2. **Selection Mechanic:** Players draw a selection rectangle over cells
3. **Scoring Condition:** Selected cells must sum to exactly 10 to score
4. **Clearing Behavior:** Scored cells disappear from the board
5. **Gap Mechanics:** After cells clear, remaining cells can be selected across the gap

## Solver Requirements

### Core Features
- Window detection and capture lock
- Grid recognition and number extraction (OCR)
- Valid move identification (combinations summing to 10)

### Optimization Goals
- Consider order of operations for maximum scoring
- Evaluate cascade opportunities created by clearing cells
- Account for gaps and new adjacencies after moves

## Development Guidelines

- Always investigate code and architecture before making changes
- Provide implementation plans and request feedback before executing
- Include example calls and parameter descriptions for CLI scripts

## Debug Screenshots

Debug screenshots are saved to `debug/` directory during overlay operation. These capture the game state at each frame.

### Example Files
- `debug/example_clean_grid.png` - Game grid without overlay interference (correct OCR)
- `debug/example_with_overlay.png` - Game grid WITH our green selection overlay visible

### Known Issue: Overlay Capture Interference

**Problem:** The screen capture (`mss`) grabs the screen region which includes our transparent overlay. When the green selection overlay is drawn over cells, the white digit text gets color-tinted, causing OCR to miss those cells (reads as 0).

**Root Cause:**
- `window_capture.py` uses `mss.grab()` which captures screen pixels, not window contents
- Our PyQt5 overlay is a separate transparent window drawn on top
- OCR uses RGB threshold (all channels > 232) - green tint causes digits to fail threshold

**Potential Fixes:**
1. Use `PrintWindow` API to capture window DC directly (excludes overlays)
2. Hide overlay before capture, restore after
3. Lower RGB threshold (risks catching game frame at ~228)

### OCR Detection Details

- White digit detection: RGB all channels > 232
- Game board frame is cream-colored (~228) - threshold must stay above this
- Grid bounds: X(650-1150), Y(230-1000)
- Expected grid: 10 columns x 16 rows
