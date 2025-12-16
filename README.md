# Nikke Math Solver

A real-time puzzle solver for the Nikke Math number grid game. Captures the game window, reads the board via OCR, calculates optimal moves, and displays solutions through a transparent overlay.

## Features

- **Window Detection** - Automatically finds and locks onto the game window
- **OCR Board Recognition** - Template-matching engine reads grid numbers (1-9)
- **Optimal Move Calculation** - Finds cell combinations that sum to exactly 10
- **Transparent Overlay** - Displays recommended moves directly on the game
- **Pluggable Architecture** - Extensible OCR engines and solver strategies

## Requirements

- Python 3.x
- Windows OS
- Target game process (default: `nikke.exe`)

## Quick Start

```bash
# Option 1: Use the batch script (recommended)
run.bat

# Option 2: Manual setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Usage

```bash
python main.py [--process PROCESS_NAME] [--debug]
```

### Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--process` | `-p` | Target process name to monitor | `nikke.exe` |
| `--debug` | `-d` | Enable debug mode (saves OCR screenshots to `debug/`) | `False` |

### Examples

```bash
# Run with default settings
python main.py

# Target a different process
python main.py --process notepad.exe

# Enable debug screenshots
python main.py --debug

# Combined
python main.py -p nikke.exe -d
```

## How It Works

```
+------------------+     +-------------------+     +------------------+
|   Control UI     |---->|   Solver Worker   |---->|  Overlay Display |
|   (PyQt5)        |     |   (Background)    |     |  (Transparent)   |
+------------------+     +-------------------+     +------------------+
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
      +-------------+   +------------+   +-------------+
      |   Window    |   |    OCR     |   |   Solver    |
      |   Capture   |   |   Engine   |   |   Strategy  |
      +-------------+   +------------+   +-------------+
```

1. **Control UI** - Start/stop solver, select strategy, view status
2. **Window Capture** - Detects game window and captures frames (10 FPS)
3. **OCR Engine** - Processes frames to extract grid numbers
4. **Solver Strategy** - Calculates optimal moves from board state
5. **Overlay Display** - Renders green rectangle over recommended cells

## Game Rules

1. **Grid Structure** - Board contains cells with numbers 1-9
2. **Selection** - Draw a rectangle over adjacent cells
3. **Scoring** - Selected cells must sum to exactly 10
4. **Clearing** - Successfully scored cells disappear
5. **Cascading** - Remaining cells can be selected across cleared gaps

## Solver Strategies

| Strategy | Description | Speed |
|----------|-------------|-------|
| **Greedy** | Single-pass, maximizes cells cleared per move | Instant (<1s) |
| **Greedy Plus** | Lookahead evaluation for better move ordering | Fast (<5s) |

Strategies are pluggable - see [docs/SOLVER_FRAMEWORK.md](docs/SOLVER_FRAMEWORK.md) for creating custom strategies.

## Project Structure

```
Nikke Math/
├── main.py                 # Application entry point
├── run.bat                 # Quick-start batch script
├── requirements.txt        # Python dependencies
├── src/
│   ├── control_ui.py       # PyQt5 control window
│   ├── window_capture.py   # Game window detection/capture
│   ├── solver_worker.py    # Background processing thread
│   ├── solution_manager.py # Move state machine
│   ├── overlay_display.py  # Transparent overlay rendering
│   ├── ocr/                # OCR engine framework
│   │   ├── base.py         # Abstract OCR engine
│   │   ├── template_engine.py  # Default template matching
│   │   └── factory.py      # Engine registration
│   └── solver/             # Solver framework
│       ├── board.py        # Board state representation
│       ├── move.py         # Move data structure
│       ├── base.py         # Abstract strategy
│       ├── factory.py      # Strategy registration
│       └── strategies/     # Strategy implementations
├── tests/                  # Unit tests
├── tools/                  # Development utilities
├── docs/                   # Documentation
└── debug/                  # Runtime debug screenshots
```

## Dependencies

- **PyQt5** - GUI framework and overlay windows
- **pywin32** - Windows API for window management
- **psutil** - Process enumeration
- **mss** - Fast screen capture
- **Pillow** - Image processing
- **opencv-python** - Template matching OCR
- **numpy** - Array operations

## Development

### Running Tests

```bash
pytest tests/
```

### Debug Tools

```bash
# OCR debugging
python tools/debug_ocr.py

# Extract templates for OCR training
python tools/extract_templates.py
```

### Extending the Application

- **Custom OCR Engine** - See [docs/ocr-engine-guide.md](docs/ocr-engine-guide.md)
- **Custom Solver Strategy** - See [docs/SOLVER_FRAMEWORK.md](docs/SOLVER_FRAMEWORK.md)
- **Full Project Plan** - See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)

## Troubleshooting

### Overlay Capture Interference

**Problem:** OCR misreads cells covered by the green overlay.

**Cause:** Screen capture includes the transparent overlay, tinting white digits green.

**Workarounds:**
- The solver uses frame stability detection to minimize this issue
- Debug mode saves screenshots for diagnosis

### OCR Detection Issues

**Problem:** Cells read as 0 or incorrect values.

**Details:**
- White digit threshold: RGB > 232 (all channels)
- Game frame color: ~228 (must stay below threshold)
- Expected grid: 10 columns x 16 rows

**Debug:** Enable `--debug` flag and check `debug/` folder for captured frames.

### Window Not Detected

**Problem:** "Window not found" status.

**Solutions:**
1. Verify the game is running
2. Check process name matches (`--process` argument)
3. Ensure game window is visible (not minimized)

## License

This project is for educational and personal use.
