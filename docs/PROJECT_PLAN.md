# Nikke Math - Project Build Plan

## Project Summary

Build a real-time puzzle solver that captures a game window, reads the board state via OCR, calculates optimal moves (cells summing to 10), and displays solutions via a transparent overlay.

---

## Current State

| Component | Status |
|-----------|--------|
| Control UI | Complete (Phase 1) |
| Window Capture | Complete (Phase 2) |
| Integration Framework | Complete (Phase 3) |
| OCR Engine | Complete (Phase 4) |
| Solver Engine | Complete (Phase 5) |
| Overlay Display | Complete (Phase 6) |

---

## Architecture Diagram

```
                              +------------------+
                              |   Control UI     |
                              |   (PyQt5 Window) |
                              |  [Start/Stop]    |
                              +--------+---------+
                                       |
                                       v
+------------------+     +------------------+     +------------------+
|  Window Capture  | --> |   OCR Engine     | --> |   Solver Engine  |
|  (pywin32/mss)   |     |  (template match)   |     |  (algorithm)     |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
+------------------+     +------------------+     +------------------+
|   Game Window    | <-- |  Overlay Layer   | <-- |  Move Generator  |
|                  |     | (PythonOverlayLib)|     |  (coordinates)   |
+------------------+     +------------------+     +------------------+
```

---

## Phase 1: Control UI Window

**Goal:** Simple Windows GUI with single Start/Stop button to control the solver stack

### UI Wireframe

```
+----------------------------------+
|  Nikke Math Solver         - X  |
+----------------------------------+
|                                  |
|  Status: Stopped                 |
|                                  |
|  +----------------------------+  |
|  |                            |  |
|  |      [ START SOLVER ]      |  |
|  |                            |  |
|  +----------------------------+  |
|                                  |
|  Window: nikke.exe not found     |
|  Board:  --                      |
|  Moves:  --                      |
|                                  |
+----------------------------------+
```

When running:
```
+----------------------------------+
|  Nikke Math Solver         - X  |
+----------------------------------+
|                                  |
|  Status: Running                 |
|                                  |
|  +----------------------------+  |
|  |                            |  |
|  |      [ STOP SOLVER ]       |  |
|  |                            |  |
|  +----------------------------+  |
|                                  |
|  Window: nikke.exe (1920x1080)   |
|  Board:  17x10 detected          |
|  Moves:  12 solutions found      |
|                                  |
+----------------------------------+
```

### Tasks
1. [x] Create `control_ui.py` module
2. [x] Create PyQt5 main window (minimal, always-on-top option)
3. [x] Implement Start/Stop toggle button
4. [x] Add status label (Stopped/Running/Error)
5. [x] Add info labels:
   - Window detection status
   - Board detection status
   - Solution count
6. [x] Define signals for worker thread communication
7. [x] Handle graceful shutdown on window close

### Dependencies
- `PyQt5` - UI framework (also required by overlay lib)

### Deliverable
A module that provides:
- `ControlWindow` class - Main UI window
- Single button toggles entire capture/solve/overlay stack
- Status feedback to user
- Signal/slot infrastructure for worker thread

---

## Phase 2: Window Capture Module

**Goal:** Lock onto `nikke.exe` process window and continuously capture it

### Target Process
- **Executable:** `nikke.exe`
- **Detection:** Find window by process name, not window title (more reliable)

### Tasks
1. [x] Create `window_capture.py` module
2. [x] Implement process enumeration (find `nikke.exe` PID)
3. [x] Get window handle (HWND) from process ID
4. [x] Implement window locking (track window position/size)
5. [x] Implement screen capture of locked region
6. [x] Handle window move/resize events
7. [x] Add error handling for process/window loss
8. [x] Auto-detect when game starts/closes

### Dependencies
- `pywin32` - Windows API access (EnumWindows, GetWindowThreadProcessId)
- `psutil` - Process enumeration (find nikke.exe)
- `mss` or `PIL.ImageGrab` - Screen capture

### Deliverable
A module that provides:
- `find_nikke_window()` - Locate nikke.exe window handle
- `is_nikke_running()` - Check if process is active
- `capture_window(hwnd)` - Return PIL Image of window contents
- `get_window_rect(hwnd)` - Return (x, y, width, height)

---

## Phase 3: Integration Framework

**Goal:** Connect UI to worker thread for testable incremental development

### Tasks
1. [x] Create `solver_worker.py` module with `SolverWorker` QThread class
2. [x] Implement worker loop with window capture integration
3. [x] Connect UI signals to worker start/stop
4. [x] Implement thread-safe status updates to UI
5. [x] Add logging infrastructure

### Threading Model
```
+----------------+          +------------------+
|   Main Thread  |  signal  |  Worker Thread   |
|   (PyQt5 UI)   | <------> | (Capture/Solve)  |
+----------------+          +------------------+
        |                            |
        v                            v
  Control Window              Status Updates
```

### Deliverable
Working integration with:
- Worker thread that monitors window capture
- Real-time status updates in UI
- Foundation for adding OCR/solver/overlay incrementally

---

## Phase 4: OCR Engine Improvements

**Goal:** Reliably extract numbers 1-9 from grid cells

### Updated UI Wireframe (Phase 4)

```
+----------------------------------+
|  Nikke Math Solver         - X  |
+----------------------------------+
|                                  |
|  Status: Running                 |
|                                  |
|  +----------------------------+  |
|  |      [ STOP SOLVER ]       |  |
|  +----------------------------+  |
|                                  |
|  Window: nikke.exe (1920x1080)   |
|  Board:  17x10 detected          |
|  OCR:    98% confidence          |
|  Cells:  170 read (2 uncertain)  |
|  FPS:    10.0 (cap: 10)          |
|  Moves:  --                      |
|                                  |
|  [ Save Debug Image ]            |
|                                  |
+----------------------------------+
```

### UI Verification Points

| Indicator | Source | Verification |
|-----------|--------|--------------|
| Board: `17x10 detected` | Grid detection | Confirms grid boundaries found |
| OCR: `98% confidence` | Average cell confidence | Shows overall OCR quality |
| Cells: `170 read (2 uncertain)` | Per-cell results | Shows cell count and problem areas |
| FPS: `10.0 (cap: 10)` | Worker loop timing | Confirms capture rate and throttling |
| Debug Image button | Manual trigger | Exports preprocessed image for troubleshooting |

### Tasks

**Core OCR Implementation:**
1. [x] Create `ocr_engine.py` module (extract from game_solver.py)
2. [x] Implement grid detection (find cell boundaries)
3. [x] Implement cell extraction with proper preprocessing
4. [x] Add image preprocessing pipeline:
   - Grayscale conversion
   - Contrast enhancement (CLAHE)
   - Noise reduction (Gaussian blur)
   - Thresholding (Adaptive)
5. [x] Implement digit recognition with confidence scoring (template matching)
6. [x] Add fallback strategies for low-confidence reads
7. [ ] Create calibration utility for new game versions (deferred)

**UI Integration & Verification:**
8. [x] Add `ocr_status` signal to worker thread (confidence %, cell counts)
9. [x] Add OCR confidence label to Control UI
10. [x] Add cell read status label to Control UI (total/uncertain)
11. [x] Add "Save Debug Image" button to Control UI
12. [x] Implement debug image export (saves preprocessed grid + annotations)
13. [x] Update board status label with grid dimensions
14. [x] Add color coding to OCR label (green >95%, yellow 80-95%, red <80%)

**Performance & Throttling:**
15. [x] Implement 10 FPS hard cap in worker loop (100ms minimum frame time)
16. [x] Add FPS counter with rolling average (last 10 frames)
17. [x] Add FPS label to Control UI showing current rate and cap
18. [x] Add `fps_update` signal to worker thread

### Dependencies
- `opencv-python` - Image preprocessing and template matching
- `numpy` - Array operations
- `Pillow` - Image manipulation

Note: Tesseract was replaced with OpenCV template matching for simpler deployment (no external install required).

### Deliverable
A module that provides:
- `detect_grid(image)` - Return grid bounds and cell positions
- `extract_board(image, grid_info)` - Return 2D array of numbers with confidence
- `preprocess_cell(cell_image)` - Return enhanced image for OCR
- `OCRResult` class - Contains board data, confidence scores, uncertain cells
- `save_debug_image(image, grid_info, results, path)` - Export annotated debug image

### Verification Checklist
- [x] OCR confidence displays and updates in real-time
- [x] Cell count matches expected grid size (160 cells for 10x16)
- [x] Debug image clearly shows grid lines and recognized digits
- [x] Color coding reflects actual OCR quality
- [x] FPS displays and never exceeds 10
- [x] FPS drops gracefully under heavy OCR load

---

## Phase 5: Solver Engine Refactoring

**Goal:** Real-time solver that shows only the next move and reacts to board changes

### Design Overview

The solver displays **only the next move** on the overlay, not the full solution sequence. When the user executes a move (or the board changes), the solver detects the change and either advances to the next planned move or recalculates if an unexpected change occurred.

### Architecture Diagram

```
+------------------+     +------------------+     +------------------+
|   OCR Engine     | --> |  BoardState      | --> |  Solver          |
|  (detects board) |     |  (change detect) |     | (lookahead alg)  |
+------------------+     +------------------+     +------------------+
                                |                         |
                                v                         v
                         SolutionManager           Next Move Only
                         (stability + stack)       (for overlay)
```

### State Machine

```
                    +------------------+
                    |   INITIALIZING   |
                    | (no board yet)   |
                    +--------+---------+
                             |
                             v  OCR detects board
                    +--------+---------+
              +---->|     SOLVING      |<----+
              |     | (calculating...) |     |
              |     +--------+---------+     |
              |              |               |
              |              v  solution ready
              |     +--------+---------+     |
              |     |   MOVE READY     |     |
              |     | (showing next)   |     |
              |     +--------+---------+     |
              |              |               |
         unexpected          |               |
          change             v  board changed (expected)
              |     +--------+---------+     |
              +-----+   MOVE EXECUTED  +-----+
                    | (pop from stack) |  stack empty?
                    +------------------+  -> recalc
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Recalculation strategy | Full recalc on unexpected change | Ensures optimal solution from new state |
| Frame stability | 3 consecutive identical frames | Filters OCR noise/flicker |
| Solver algorithm | Lookahead (depth=2) | Smarter move ordering, acceptable performance |

### Module Structure: `src/solver.py`

```
solver.py
|
+-- BoardState (dataclass)
|   |-- board: tuple[tuple[int|None]]  # Immutable, hashable
|   |-- rows, cols: int
|   |-- __hash__(), __eq__()           # Change detection
|   |-- diff(other) -> set[(row,col)]  # Changed cells
|   |-- apply_move(move) -> BoardState # Returns new state
|   +-- from_ocr(board_2d) -> BoardState
|
+-- Move (dataclass)
|   |-- r1, c1, r2, c2: int            # Rectangle bounds
|   |-- cells: list[(row, col, value)] # Cells being cleared
|   |-- total: int                     # Sum (always 10)
|   +-- screen_rect: optional          # For overlay (set later)
|
+-- Solver (class)
|   |-- find_all_valid_moves(board) -> list[Move]
|   |-- solve_lookahead(board, depth=2) -> list[Move]
|   +-- get_next_move(board) -> Move | None
|
+-- SolutionManager (class)
    |-- STABILITY_THRESHOLD = 3
    |-- _solution_stack: list[Move]
    |-- _last_stable_board: BoardState | None
    |-- _pending_board: BoardState | None
    |-- _stable_frame_count: int
    |
    |-- update(new_board: BoardState) -> Move | None
    |   1. Check if new_board matches _pending_board
    |   2. If yes, increment _stable_frame_count
    |   3. If no, reset _pending_board and counter
    |   4. When stable (3 frames):
    |      - If unchanged from _last_stable_board -> return top of stack
    |      - If matches expected post-move -> pop stack, advance
    |      - If unexpected change -> full recalc with lookahead
    |   5. Return next move (top of stack) or None
    |
    +-- reset() -> None  # Clear stack and state
```

### Tasks

**Core Data Structures:**
1. [x] Create `src/solver.py` module
2. [x] Implement `BoardState` dataclass:
   - Immutable tuple-of-tuples representation
   - `__hash__()` and `__eq__()` for change detection
   - `diff(other)` method returning changed cell positions
   - `apply_move(move)` returning new BoardState with cells cleared
   - `from_ocr(board_2d)` factory method
3. [x] Implement `Move` dataclass:
   - Rectangle bounds (r1, c1, r2, c2)
   - Cell list with positions and values
   - Total sum (validation)
   - Optional screen coordinates for overlay

**Solver Algorithm:**
4. [x] Port `find_all_rectangles_summing_to_10()` from game_solver.py as `find_all_valid_moves()`
5. [x] Port `solve_lookahead()` from game_solver.py (depth=2 default)
6. [x] Add `get_next_move()` convenience method (returns first move from solution)

**Solution Management:**
7. [x] Implement `SolutionManager` class:
   - Frame stability tracking (3 consecutive frames)
   - Solution stack management
   - Expected vs unexpected change detection
   - Automatic recalculation on unexpected changes
8. [x] Add `update(board)` method with full state machine logic
9. [x] Add `reset()` method for manual clearing

**Worker Integration:**
10. [x] Add `next_move_ready` signal to `SolverWorker`: `pyqtSignal(object)` emitting Move or None
11. [x] Instantiate `SolutionManager` in worker `__init__`
12. [x] Integrate into `_process_cycle()`:
    - Convert OCR result to BoardState
    - Call `solution_manager.update(board_state)`
    - Emit `next_move_ready` signal with result
    - Update `moves_changed` status text
13. [x] Handle solver reset on worker stop/start

**Testing & Validation:**
14. [x] Unit tests for BoardState hash/equality
15. [x] Unit tests for Move creation and validation
16. [x] Unit tests for find_all_valid_moves correctness
17. [x] Integration test: stability threshold behavior
18. [ ] Integration test: expected move detection and stack pop

### Dependencies
- `numpy` - Array operations (already installed)
- No new dependencies required

### Deliverable
A module that provides:
- `BoardState` class - Immutable, hashable board with change detection
- `Move` class - Single move representation with cell details
- `Solver` class - Lookahead algorithm returning ordered move list
- `SolutionManager` class - Real-time state tracking with stability and stack
- `next_move_ready` signal - Emits single Move for overlay consumption

### Verification Checklist
- [x] BoardState correctly detects identical vs changed boards
- [x] 3-frame stability filters single-frame OCR glitches
- [ ] Expected move execution pops from stack without recalc
- [x] Unexpected changes trigger full recalculation
- [x] UI shows "1 move ready" when solution available
- [x] UI shows "No valid moves" when board is solved or stuck
- [x] Worker cleanly resets solver state on stop/start

---

## Phase 6: Overlay Display System

**Goal:** Display the next recommended move as a transparent rectangle overlay on the game window

### Design Overview

The overlay shows **only the next move** as a highlighted rectangle. Since the solver already computes lookahead solutions, we display one move at a time. When the user executes the move (board changes), the next move in the solution automatically displays.

### Architecture Diagram

```
+------------------+     +-------------------+     +------------------+
|  SolverWorker    | --> |  OverlayManager   | --> | PythonOverlayLib |
|  next_move_ready |     |  (coord mapping)  |     |   (rendering)    |
|  grid_info_ready |     |  (thread-safe)    |     |                  |
+------------------+     +-------------------+     +------------------+
         |                        |
         v                        v
   Move (r1,c1,r2,c2)      Screen coordinates
   GridInfo (positions)    for rectangle drawing
```

### Visual Design

```
Game Window with Overlay:
+------------------------------------------+
|                                          |
|    [1] [2] [3] [4] [5] [6] [7] [8]       |
|    [9] [1] [2] [3] [4] [5] [6] [7]       |
|    +===========================+         |
|    || [8] [9] [1] [2] ||  <-- Green     |
|    +===========================+   rectangle
|    [3] [4] [5] [6] [7] [8] [9] [1]       |
|                                          |
+------------------------------------------+

Move indicator shows:
- Semi-transparent green fill
- Bright green border (3px)
- Covers entire selection rectangle
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Threading model | Shared state with lock | PythonOverlayLib runs its own event loop; use threading.Lock for move updates |
| Coordinate source | GridInfo.cell_positions | Already computed by OCR, includes exact pixel positions |
| Visual style | Green rectangle, no text | Simple, clear, non-intrusive |
| Refresh rate | 60 FPS (16ms) | Smooth visuals, standard for overlays |
| Window tracking | Poll window position | Update overlay position when game window moves |

### Module Structure: `src/overlay_display.py`

```
overlay_display.py
|
+-- OverlayManager (class)
|   |-- _overlay: Overlay              # PythonOverlayLib instance
|   |-- _current_move: Move | None     # Thread-safe move storage
|   |-- _grid_info: GridInfo | None    # Current grid info for mapping
|   |-- _window_rect: tuple | None     # Game window screen position
|   |-- _lock: threading.Lock          # Thread safety
|   |-- _running: bool                 # Overlay active flag
|   |
|   |-- start() -> None                # Start overlay (runs in thread)
|   |-- stop() -> None                 # Stop and cleanup overlay
|   |-- update_move(move, grid_info, window_rect) -> None
|   |   # Thread-safe update of current move to display
|   |-- clear_move() -> None           # Clear current move display
|   |
|   +-- _draw_callback() -> list[Shape]
|       # Returns shapes for PythonOverlayLib to render
|
+-- Helper Functions
    |-- grid_to_screen(row, col, grid_info, window_rect) -> (x, y)
    |-- get_move_screen_rect(move, grid_info, window_rect) -> (x, y, w, h)
```

### Tasks

**Core Module Setup:**
1. [x] Create `src/overlay_display.py` module
2. [x] Add PythonOverlayLib imports and color constants:
   - `HIGHLIGHT_FILL`: Semi-transparent green (0, 255, 0, 80)
   - `HIGHLIGHT_BORDER`: Bright green (0, 255, 0, 220)
   - `BORDER_THICKNESS`: 3px

**Coordinate Mapping:**
3. [x] Implement `grid_to_screen(row, col, grid_info, window_rect)`:
   - Use GridInfo.cell_positions for cell center coordinates
   - Add window_rect offset for absolute screen position
4. [x] Implement `get_move_screen_rect(move, grid_info, window_rect)`:
   - Calculate bounding box from move's r1,c1 to r2,c2
   - Return (x, y, width, height) in screen coordinates

**OverlayManager Class:**
5. [x] Implement `__init__()`:
   - Initialize threading.Lock
   - Initialize state variables (move, grid_info, window_rect)
   - Do NOT create Overlay yet (created in start())
6. [x] Implement `start()`:
   - Create Overlay with draw callback
   - Run in background thread (Overlay.spawn() blocks)
7. [x] Implement `stop()`:
   - Set _running = False
   - Clean shutdown of overlay
8. [x] Implement `update_move(move, grid_info, window_rect)`:
   - Acquire lock, update state, release lock
   - Thread-safe for worker thread calls
9. [x] Implement `clear_move()`:
   - Set _current_move = None
10. [x] Implement `_draw_callback()`:
    - Acquire lock, read current state
    - If move exists, create FlDrawRect (fill) + SkDrawRect (border)
    - Return list of shapes

**Worker Integration:**
11. [x] Add `overlay_update` signal to SolverWorker: `pyqtSignal(object, object, object)`
12. [x] Emit grid_info along with move in worker cycle
13. [x] Update main.py to:
    - Create OverlayManager instance
    - Connect worker signals to overlay updates
    - Start/stop overlay with solver
    - Pass window_rect from WindowCapture

**Window Tracking:**
14. [x] Window rect included in overlay_update signal (emitted every frame)
15. [x] Update overlay position when game window moves (automatic via signal)
16. [ ] Handle window minimize/restore (hide/show overlay) - deferred to Phase 7

**Testing & Edge Cases:**
17. [ ] Test overlay appears over game window
18. [ ] Test overlay updates when move changes
19. [ ] Test overlay clears when no valid moves
20. [ ] Test overlay tracks window movement
21. [ ] Test clean shutdown (no orphan overlay)

### Dependencies
- `PythonOverlayLib` - Overlay rendering (already in requirements)
- `PyQt5` - Required by overlay lib (already installed)
- `threading` - Standard library, for Lock

### Deliverable
A module that provides:
- `OverlayManager` class - Manages overlay lifecycle and rendering
- `update_move(move, grid_info, window_rect)` - Thread-safe move update
- `clear_move()` - Hide move display
- `start()` / `stop()` - Overlay lifecycle control

### Verification Checklist
- [ ] Green rectangle appears over correct cells
- [ ] Rectangle updates when next move changes
- [ ] Rectangle clears when no moves available
- [ ] Overlay tracks game window position
- [ ] Overlay starts/stops with solver toggle
- [ ] No performance impact on main UI
- [ ] Clean shutdown with no orphan windows

---

## Phase 7: Polish and Optimization

**Goal:** Production-ready application

### Tasks
1. [ ] Performance profiling and optimization
2. [ ] Error handling improvements
3. [ ] User feedback (audio/visual cues)
4. [ ] Settings persistence
5. [ ] Documentation:
   - README with setup instructions
   - Usage guide
   - Troubleshooting guide
6. [ ] Create `requirements.txt`
7. [ ] Add unit tests for solver
8. [ ] Add integration tests

---

## File Structure (Target)

```
Nikke Math/
|-- main.py                 # Entry point
|-- requirements.txt        # Dependencies
|-- config.yaml            # Configuration
|
|-- src/
|   |-- __init__.py
|   |-- control_ui.py      # Phase 1
|   |-- window_capture.py  # Phase 2
|   |-- ocr_engine.py      # Phase 3
|   |-- solver.py          # Phase 4
|   |-- overlay_display.py # Phase 5
|   |-- solver_worker.py   # Phase 6 - Background thread
|   |-- board_state.py     # Board data structures
|   |-- utils.py           # Shared utilities
|
|-- tests/
|   |-- test_solver.py
|   |-- test_ocr.py
|   |-- test_integration.py
|
|-- docs/
|   |-- PROJECT_PLAN.md    # This file
|   |-- overlaylib-api.md  # Overlay library reference
|   |-- USAGE.md           # User guide
|
|-- assets/
|   |-- test_images/       # Sample screenshots for testing
```

---

## Dependencies Summary

```
# requirements.txt
PyQt5>=5.15           # Control UI + Overlay lib
pywin32>=306          # Windows API
psutil>=5.9           # Process enumeration (find nikke.exe)
mss>=9.0              # Screen capture
Pillow>=10.0          # Image processing
opencv-python>=4.8    # Image preprocessing and template matching OCR
numpy>=1.24           # Array operations
```

Note: Tesseract and EasyOCR dependencies removed - using OpenCV template matching for simpler deployment.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OCR accuracy issues | High | High | Multiple preprocessing strategies, confidence thresholds |
| Game window detection fails | Medium | High | Manual window selection fallback |
| Overlay performance | Medium | Medium | Reduce refresh rate, optimize drawing |
| Game updates break OCR | Medium | High | Calibration utility, configurable parameters |

---

## Success Criteria

1. Window capture locks reliably to game
2. OCR achieves >95% accuracy on board state
3. Solver finds solution in <1 second
4. Overlay displays clearly without obscuring gameplay
5. Full cycle (capture -> solve -> display) completes in <2 seconds

---

## Next Steps

1. Review this plan and provide feedback
2. Set up project structure (folders, requirements.txt)
3. Begin Phase 1: Control UI Window
