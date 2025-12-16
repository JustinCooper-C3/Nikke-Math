# Solver Framework Specification

## Overview

A modular solution framework for the Nikke Math puzzle solver that supports multiple solving strategies with runtime selection, cancellation, and progress tracking.

---

## Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                 UI LAYER                                           |
+-----------------------------------------------------------------------------------+
|  ControlWindow                                                                     |
|  +---------------------------+  +------------------+  +------------------------+   |
|  | [Start] [Stop]            |  | Strategy: [v]    |  | Status: Computing...   |   |
|  +---------------------------+  | - Greedy         |  | Progress: 45%          |   |
|                                 | - Lookahead      |  | Time: 3.2s / 20s       |   |
|                                 | - Exhaustive     |  +------------------------+   |
|                                 +------------------+                               |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                            SOLVER WORKER                                           |
+-----------------------------------------------------------------------------------+
|  SolverWorker (QThread)                                                            |
|  - strategy_name: str              # Currently selected strategy                   |
|  - cancel_flag: threading.Event    # Set by Stop button                            |
|  - current_context: SolutionContext                                                |
|                                                                                    |
|  on_strategy_changed(name):        # Slot for UI dropdown                          |
|      self.strategy_name = name                                                     |
|      self.solution_manager.reset() # Restart with new strategy                     |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                         STRATEGY FRAMEWORK                                         |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  +------------------+       +------------------------------------------------+    |
|  | SolutionContext  |       |              SolverStrategy (ABC)              |    |
|  +------------------+       +------------------------------------------------+    |
|  | board: BoardState|       | + name: str                                    |    |
|  | cancel: Event    | ----> | + timeout_sec: float = 20.0                    |    |
|  | timeout: float   |       | + solve(context) -> Solution                   |    |
|  | progress_cb      |       | + _check_cancelled(context) -> bool            |    |
|  +------------------+       +------------------------------------------------+    |
|                                            ^                                       |
|                                            |                                       |
|            +---------------+---------------+---------------+                       |
|            |               |               |               |                       |
|     +------+-----+  +------+------+  +-----+-------+  +----+--------+             |
|     |   Greedy   |  |  Lookahead  |  | Exhaustive  |  |   Genetic   |             |
|     +------------+  +-------------+  +-------------+  +-------------+             |
|     | Instant    |  | depth=3     |  | DFS+memo    |  | population  |             |
|     | No timeout |  | beam=10     |  | pruning     |  | generations |             |
|     +------------+  +-------------+  +-------------+  +-------------+             |
|                                                                                    |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                              SOLUTION                                              |
+-----------------------------------------------------------------------------------+
|  + moves: List[Move]                                                               |
|  + total_cleared: int                                                              |
|  + is_complete: bool          # True if no more valid moves                        |
|  + was_cancelled: bool        # True if stopped mid-computation                    |
|  + metrics: SolutionMetrics   # time, states explored, strategy name               |
+-----------------------------------------------------------------------------------+
```

---

## Module Structure

```
src/
├── solver/
│   ├── __init__.py           # Public API: create_strategy(), STRATEGIES
│   ├── board.py              # BoardState (extracted from current solver.py)
│   ├── move.py               # Move dataclass
│   ├── context.py            # SolutionContext with cancel_flag + progress
│   ├── solution.py           # Solution + SolutionMetrics
│   ├── base.py               # SolverStrategy ABC with timeout/cancel support
│   ├── factory.py            # Registry: create_strategy(name) -> Strategy
│   └── strategies/
│       ├── __init__.py
│       ├── greedy.py         # Immediate: max cells per move
│       ├── lookahead.py      # Beam search (depth=3, beam=10)
│       └── exhaustive.py     # Full DFS with memoization + pruning
│
├── control_ui.py             # Add strategy dropdown + progress display
├── solver_worker.py          # Add strategy switching + cancel support
└── ... (existing files)
```

---

## Core Components

### BoardState (`board.py`)

Immutable representation of the game board.

```python
@dataclass(frozen=True)
class BoardState:
    grid: Tuple[Tuple[Optional[int], ...], ...]
    rows: int
    cols: int

    @classmethod
    def from_2d_list(cls, grid: List[List[Optional[int]]]) -> "BoardState"

    @classmethod
    def from_ocr(cls, ocr_result: OCRResult) -> "BoardState"

    def apply_move(self, move: Move) -> "BoardState"
    def get_cell(self, row: int, col: int) -> Optional[int]
    def count_cells(self) -> int
    def __hash__(self) -> int  # For memoization
```

### Move (`move.py`)

Represents a single valid puzzle move (rectangle selection summing to 10).

```python
@dataclass(frozen=True)
class Move:
    r1: int  # Top row (inclusive)
    c1: int  # Left column (inclusive)
    r2: int  # Bottom row (inclusive)
    c2: int  # Right column (inclusive)
    cells: Tuple[Tuple[int, int], ...]  # Non-empty cells in rectangle
    total: int  # Always 10

    @property
    def cell_count(self) -> int

    @property
    def area(self) -> int
```

### SolutionContext (`context.py`)

Shared context passed to strategies containing board state, cancellation, and progress reporting.

```python
@dataclass
class SolutionContext:
    board: BoardState
    cancel_flag: threading.Event
    timeout_sec: float = 20.0
    start_time: float = field(default_factory=time.time)
    progress_callback: Optional[Callable[[float, str], None]] = None

    def is_cancelled(self) -> bool:
        """Check if cancellation requested or timeout exceeded."""
        if self.cancel_flag.is_set():
            return True
        if time.time() - self.start_time > self.timeout_sec:
            return True
        return False

    def report_progress(self, percent: float, message: str = "") -> None:
        """Report progress to UI (0.0 to 1.0)."""
        if self.progress_callback:
            self.progress_callback(percent, message)

    def elapsed_time(self) -> float:
        """Seconds elapsed since computation started."""
        return time.time() - self.start_time
```

### Solution (`solution.py`)

Result of a strategy computation.

```python
@dataclass
class SolutionMetrics:
    computation_time_ms: float
    states_explored: int
    pruned_branches: int
    strategy_name: str

@dataclass
class Solution:
    moves: List[Move]
    total_cleared: int
    is_complete: bool      # True if no more valid moves remain
    was_cancelled: bool    # True if stopped before completion
    metrics: SolutionMetrics
    board_states: List[BoardState]  # State after each move

    @property
    def move_count(self) -> int:
        return len(self.moves)
```

### SolverStrategy (`base.py`)

Abstract base class for all solving strategies.

```python
from abc import ABC, abstractmethod

class SolverStrategy(ABC):
    name: str = "base"
    description: str = "Base strategy"
    timeout_sec: float = 20.0

    @abstractmethod
    def solve(self, context: SolutionContext) -> Solution:
        """
        Compute solution for the given board state.

        Must periodically check context.is_cancelled() and return
        partial solution if True.
        """
        pass

    def find_all_valid_moves(self, board: BoardState) -> List[Move]:
        """Find all rectangles that sum to 10."""
        # Shared implementation for all strategies
        pass

    def _check_cancelled(self, context: SolutionContext) -> bool:
        """Convenience method to check cancellation."""
        return context.is_cancelled()
```

### Strategy Factory (`factory.py`)

Registry and factory for strategy instantiation.

```python
STRATEGIES: Dict[str, Type[SolverStrategy]] = {}

def register_strategy(cls: Type[SolverStrategy]) -> Type[SolverStrategy]:
    """Decorator to register a strategy."""
    STRATEGIES[cls.name] = cls
    return cls

def create_strategy(name: str, **kwargs) -> SolverStrategy:
    """Create strategy instance by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGIES[name](**kwargs)

def get_strategy_names() -> List[str]:
    """Get list of available strategy names."""
    return list(STRATEGIES.keys())

def get_strategy_info() -> List[Dict[str, str]]:
    """Get name and description for all strategies."""
    return [
        {"name": cls.name, "description": cls.description}
        for cls in STRATEGIES.values()
    ]
```

---

## Strategy Implementations

### GreedyStrategy (`strategies/greedy.py`)

Immediate, single-pass strategy that always picks the move clearing the most cells.

```python
@register_strategy
class GreedyStrategy(SolverStrategy):
    name = "greedy"
    description = "Greedy (instant) - Always picks move clearing most cells"
    timeout_sec = 1.0  # Effectively instant

    def solve(self, context: SolutionContext) -> Solution:
        board = context.board
        moves = []
        board_states = [board]
        states_explored = 0

        while True:
            if self._check_cancelled(context):
                return self._build_solution(moves, board_states, states_explored,
                                           was_cancelled=True)

            valid_moves = self.find_all_valid_moves(board)
            states_explored += len(valid_moves)

            if not valid_moves:
                break

            # Pick move with most cells
            best_move = max(valid_moves, key=lambda m: m.cell_count)
            moves.append(best_move)
            board = board.apply_move(best_move)
            board_states.append(board)

            context.report_progress(len(moves) / 50, f"{len(moves)} moves found")

        return self._build_solution(moves, board_states, states_explored,
                                   was_cancelled=False)
```

### LookaheadStrategy (`strategies/lookahead.py`)

Beam search with configurable depth and beam width.

```python
@register_strategy
class LookaheadStrategy(SolverStrategy):
    name = "lookahead"
    description = "Lookahead (fast) - Beam search with depth 3"
    timeout_sec = 5.0

    def __init__(self, depth: int = 3, beam_width: int = 10):
        self.depth = depth
        self.beam_width = beam_width

    def solve(self, context: SolutionContext) -> Solution:
        """
        Beam search: at each step, evaluate top beam_width moves
        by looking ahead depth moves.
        """
        board = context.board
        moves = []
        board_states = [board]
        states_explored = 0

        while True:
            if self._check_cancelled(context):
                return self._build_solution(moves, board_states, states_explored,
                                           was_cancelled=True)

            valid_moves = self.find_all_valid_moves(board)
            if not valid_moves:
                break

            # Score each move by lookahead
            scored_moves = []
            for move in valid_moves[:self.beam_width]:
                if self._check_cancelled(context):
                    break
                score, explored = self._lookahead_score(
                    board.apply_move(move),
                    self.depth - 1,
                    context
                )
                scored_moves.append((move, score + move.cell_count))
                states_explored += explored

            if not scored_moves:
                break

            best_move = max(scored_moves, key=lambda x: x[1])[0]
            moves.append(best_move)
            board = board.apply_move(best_move)
            board_states.append(board)

            progress = min(0.95, context.elapsed_time() / self.timeout_sec)
            context.report_progress(progress, f"{len(moves)} moves, {states_explored} states")

        return self._build_solution(moves, board_states, states_explored,
                                   was_cancelled=False)

    def _lookahead_score(self, board: BoardState, depth: int,
                         context: SolutionContext) -> Tuple[int, int]:
        """Recursively score board state by cells clearable in depth moves."""
        if depth == 0 or self._check_cancelled(context):
            return 0, 1

        valid_moves = self.find_all_valid_moves(board)
        if not valid_moves:
            return 0, 1

        best_score = 0
        total_explored = len(valid_moves)

        for move in valid_moves[:self.beam_width]:
            future_score, explored = self._lookahead_score(
                board.apply_move(move),
                depth - 1,
                context
            )
            total_explored += explored
            best_score = max(best_score, move.cell_count + future_score)

        return best_score, total_explored
```

### ExhaustiveStrategy (`strategies/exhaustive.py`)

Full search with memoization and pruning for optimal solutions.

```python
@register_strategy
class ExhaustiveStrategy(SolverStrategy):
    name = "exhaustive"
    description = "Exhaustive (optimal) - Full search with pruning"
    timeout_sec = 20.0

    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        self._memo: Dict[BoardState, Tuple[List[Move], int]] = {}

    def solve(self, context: SolutionContext) -> Solution:
        """
        DFS with memoization to find optimal solution.
        Uses branch pruning based on best known solution.
        """
        self._memo.clear()
        self._best_solution: List[Move] = []
        self._best_cleared: int = 0
        self._states_explored: int = 0
        self._pruned: int = 0

        self._dfs(context.board, [], 0, context)

        # Reconstruct board states
        board_states = [context.board]
        board = context.board
        for move in self._best_solution:
            board = board.apply_move(move)
            board_states.append(board)

        return Solution(
            moves=self._best_solution,
            total_cleared=self._best_cleared,
            is_complete=not context.is_cancelled(),
            was_cancelled=context.is_cancelled(),
            board_states=board_states,
            metrics=SolutionMetrics(
                computation_time_ms=context.elapsed_time() * 1000,
                states_explored=self._states_explored,
                pruned_branches=self._pruned,
                strategy_name=self.name
            )
        )

    def _dfs(self, board: BoardState, moves: List[Move],
             cleared: int, context: SolutionContext) -> None:
        """Recursive DFS with pruning."""
        if self._check_cancelled(context):
            return

        self._states_explored += 1

        # Update progress periodically
        if self._states_explored % 1000 == 0:
            progress = min(0.95, context.elapsed_time() / self.timeout_sec)
            context.report_progress(
                progress,
                f"{self._states_explored} states, best: {self._best_cleared} cells"
            )

        # Check memo
        if board in self._memo:
            cached_moves, cached_cleared = self._memo[board]
            total_cleared = cleared + cached_cleared
            if total_cleared > self._best_cleared:
                self._best_cleared = total_cleared
                self._best_solution = moves + cached_moves
            return

        valid_moves = self.find_all_valid_moves(board)

        if not valid_moves:
            # Terminal state
            if cleared > self._best_cleared:
                self._best_cleared = cleared
                self._best_solution = moves.copy()
            self._memo[board] = ([], 0)
            return

        # Pruning: upper bound check
        max_possible = cleared + sum(m.cell_count for m in valid_moves)
        if max_possible <= self._best_cleared:
            self._pruned += 1
            return

        # Sort moves by cell_count descending (better pruning)
        valid_moves.sort(key=lambda m: m.cell_count, reverse=True)

        best_from_here: List[Move] = []
        best_cleared_from_here: int = 0

        for move in valid_moves:
            new_board = board.apply_move(move)
            new_moves = moves + [move]
            new_cleared = cleared + move.cell_count

            self._dfs(new_board, new_moves, new_cleared, context)

            # Track best from this state for memo
            if new_cleared + (self._best_cleared - cleared) > best_cleared_from_here:
                # This is approximate; full tracking would need more bookkeeping
                pass

        # Simplified memo (just store empty since we update global best)
        self._memo[board] = ([], 0)
```

---

## UI Integration

### ControlWindow Changes (`control_ui.py`)

```
+--------------------------------------------------+
|  Nikke Math Solver                          [X]  |
+--------------------------------------------------+
|  Window: nikke.exe (1920x1080)                   |
|  Board: 10x16 detected                           |
|  OCR: 98.5% confidence                           |
+--------------------------------------------------+
|  Strategy: [ Greedy          v ]                 |  <-- Dropdown (QComboBox)
|            [ ] Greedy (instant)                  |
|            [ ] Lookahead (fast)                  |
|            [ ] Exhaustive (optimal)              |
+--------------------------------------------------+
|  Solution: Ready (12 moves, 47 cells)            |  <-- Solution status
|  Progress: [=========>         ] 80%  4.2s/20s   |  <-- Progress bar
+--------------------------------------------------+
|        [ START ]        [ STOP ]                 |
+--------------------------------------------------+
```

#### New Signals

```python
class ControlWindow(QWidget):
    # Existing signals
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    # New signals
    strategy_changed = pyqtSignal(str)  # Emits strategy name
```

#### New Widgets

```python
# Strategy dropdown
self.strategy_combo = QComboBox()
for info in get_strategy_info():
    self.strategy_combo.addItem(info["description"], info["name"])
self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)

# Progress bar
self.progress_bar = QProgressBar()
self.progress_bar.setRange(0, 100)
self.progress_bar.setValue(0)

# Time label
self.time_label = QLabel("0.0s / 20.0s")
```

### SolverWorker Changes (`solver_worker.py`)

```python
class SolverWorker(QThread):
    # New signals
    progress_update = pyqtSignal(float, str)  # percent, message
    solution_ready = pyqtSignal(object)  # Solution object

    def __init__(self):
        self._strategy_name = "greedy"
        self._cancel_flag = threading.Event()
        self._strategy = create_strategy(self._strategy_name)

    @pyqtSlot(str)
    def on_strategy_changed(self, name: str):
        """Handle strategy change from UI."""
        self._strategy_name = name
        self._strategy = create_strategy(name)
        self._solution_manager.reset()

    def stop(self):
        """Stop current computation."""
        self._cancel_flag.set()
        self._running = False

    def _compute_solution(self, board: BoardState) -> Solution:
        """Run selected strategy with cancellation support."""
        self._cancel_flag.clear()

        context = SolutionContext(
            board=board,
            cancel_flag=self._cancel_flag,
            timeout_sec=self._strategy.timeout_sec,
            progress_callback=lambda p, m: self.progress_update.emit(p, m)
        )

        return self._strategy.solve(context)
```

---

## Cancellation Flow

```
User clicks STOP
       |
       v
ControlWindow.stop_requested.emit()
       |
       v
SolverWorker.stop()
       |
       +---> cancel_flag.set()
       |
       v
Strategy.solve() loop checks:
       |
       if context.is_cancelled():
           return Solution(was_cancelled=True, ...)
       |
       v
Worker emits status_changed("Stopped")
       |
       v
UI updates:
   - Progress bar resets
   - Strategy dropdown re-enabled
   - Status shows "Cancelled"
```

---

## Implementation Tasks

| # | Task | Files | Priority |
|---|------|-------|----------|
| 1 | Create `solver/` package structure | `src/solver/__init__.py` | High |
| 2 | Extract `BoardState` to module | `src/solver/board.py` | High |
| 3 | Extract `Move` to module | `src/solver/move.py` | High |
| 4 | Create `SolutionContext` | `src/solver/context.py` | High |
| 5 | Create `Solution` + `SolutionMetrics` | `src/solver/solution.py` | High |
| 6 | Create `SolverStrategy` ABC | `src/solver/base.py` | High |
| 7 | Create strategy factory/registry | `src/solver/factory.py` | High |
| 8 | Implement `GreedyStrategy` | `src/solver/strategies/greedy.py` | High |
| 9 | Implement `LookaheadStrategy` | `src/solver/strategies/lookahead.py` | Medium |
| 10 | Implement `ExhaustiveStrategy` | `src/solver/strategies/exhaustive.py` | Medium |
| 11 | Add strategy dropdown to UI | `src/control_ui.py` | High |
| 12 | Add progress bar to UI | `src/control_ui.py` | Medium |
| 13 | Integrate cancel flag in worker | `src/solver_worker.py` | High |
| 14 | Wire strategy selection signals | `src/solver_worker.py`, `src/control_ui.py` | High |
| 15 | Update `SolutionManager` for new framework | `src/solver_worker.py` | Medium |
| 16 | Migrate existing solver code | `src/solver.py` -> `src/solver/` | High |

---

## Configuration

### Strategy Defaults

| Strategy | Timeout | Parameters |
|----------|---------|------------|
| Greedy | 1s | None |
| Lookahead | 5s | depth=3, beam_width=10 |
| Exhaustive | 20s | max_depth=100 |

### Global Settings

```python
DEFAULT_STRATEGY = "greedy"
MAX_TIMEOUT_SEC = 20.0
PROGRESS_UPDATE_INTERVAL_MS = 100
```

---

## Future Extensions

### Potential Additional Strategies

- **Genetic Algorithm**: Evolutionary optimization with crossover/mutation
- **Monte Carlo Tree Search**: Random sampling with UCB selection
- **A* Search**: Heuristic-guided pathfinding
- **Hybrid**: Greedy for early moves, exhaustive for endgame

### Potential Enhancements

- Strategy parameter tuning UI
- Solution comparison mode
- Solution playback/stepping
- Export solutions to file
- Benchmark mode for strategy comparison
