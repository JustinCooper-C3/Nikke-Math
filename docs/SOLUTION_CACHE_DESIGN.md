# Solution Cache Design

## Overview

This document outlines the design for caching full solutions from the greedy_plus optimizer, enabling move-by-move playback with validation and intelligent cache invalidation.

## Problem Statement

**Current Behavior:**
- System computes ONE move at a time
- After user clears cells, system recomputes next move from scratch
- No memory of the full solution path

**Desired Behavior:**
- Compute full solution upfront (all moves)
- Pop moves sequentially as user executes them
- Validate user selected correct cells
- Only recompute if board diverges from expected state

**Challenge:**
- Board state can "flutter" when user is actively selecting cells
- OCR may misread highlighted/selected cells
- Must distinguish transient flutter from actual board changes

---

## Current Architecture

### State Machine

```
WAITING_STABLE ──> COMPUTING_MOVE ──> MOVE_DISPLAYED ──> VALIDATING_CLEAR
       ^                                                        │
       └────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `SolutionManager` | State machine for move lifecycle |
| `SolverStrategy.find_best_move()` | Computes single best move |
| `_frame_buffer` | 3-frame stability detection |
| `_unstable_cells` | Tracks flickering cells |
| `_expected_cleared` | Cells expected to disappear |

### Limitations

1. Single-move computation per cycle
2. No solution path memory
3. Full recompute after every clear

---

## Proposed Design

### Solution Cache Structure

```python
@dataclass
class CachedSolution:
    """Full solution with move queue and expected board states."""

    # The complete solution from optimizer
    solution: Solution

    # Current position in move sequence
    move_index: int = 0

    # Board state expected after each move (indexed by move_index)
    # Index 0 = initial board, Index N = board after move N-1
    expected_boards: List[BoardState]

    # Timestamp for cache staleness detection
    created_at: float

    @property
    def current_move(self) -> Optional[Move]:
        """Get next move to display."""
        if self.move_index < len(self.solution.moves):
            return self.solution.moves[self.move_index]
        return None

    @property
    def expected_board_after_current(self) -> Optional[BoardState]:
        """Get expected board state after current move executes."""
        if self.move_index < len(self.expected_boards) - 1:
            return self.expected_boards[self.move_index + 1]
        return None

    @property
    def is_exhausted(self) -> bool:
        """True if all moves have been consumed."""
        return self.move_index >= len(self.solution.moves)

    def advance(self) -> None:
        """Move to next move in sequence."""
        self.move_index += 1

    def invalidate_from(self, index: int) -> None:
        """Invalidate cache from given index onwards."""
        # Truncate solution to valid portion
        self.solution.moves = self.solution.moves[:index]
        self.expected_boards = self.expected_boards[:index + 1]
```

### Updated State Machine

```
                                    ┌─────────────────────┐
                                    │                     │
                                    v                     │
WAITING_STABLE ──> COMPUTING_SOLUTION ──> MOVE_DISPLAYED ─┘
       ^                                        │
       │                                        v
       │                               VALIDATING_CLEAR
       │                                        │
       │            ┌───────────────────────────┤
       │            │                           │
       │            v                           v
       │    [Cache Valid]                [Cache Invalid]
       │    Pop next move                Recompute solution
       │            │                           │
       │            v                           │
       └────────────┴───────────────────────────┘
```

### New States

| State | Description |
|-------|-------------|
| `WAITING_STABLE` | Collecting frames for initial stability |
| `COMPUTING_SOLUTION` | Computing FULL solution (replaces `COMPUTING_MOVE`) |
| `MOVE_DISPLAYED` | Current move shown, waiting for user action |
| `VALIDATING_CLEAR` | Checking if board matches expected state |

---

## Validation Logic

### Tiered Validation Approach

```
Tier 1 (Fast Path):
  - Check if expected cells are now empty
  - If YES → Advance cache, display next move

Tier 2 (Stability Wait):
  - If Tier 1 fails, wait for 3 stable frames
  - Filters out selection flutter

Tier 3 (Full Comparison):
  - Compare stable board to expected board
  - If MATCH → Advance cache, display next move
  - If MISMATCH → Invalidate cache, recompute
```

### Validation Flow Diagram

```
┌──────────────────────────┐
│   Board Change Detected  │
└────────────┬─────────────┘
             v
┌──────────────────────────────────────┐
│ Tier 1: Expected cells now empty?    │
│ (Check only cells in current move)   │
└────────────┬─────────────────────────┘
             │
        Yes  │  No
    ┌────────┴────────┐
    v                 v
┌────────┐    ┌──────────────────────────┐
│ VALID  │    │ Tier 2: Wait 3 stable    │
│ Advance│    │ frames (filter flutter)  │
│ Cache  │    └────────────┬─────────────┘
└────────┘                 v
                  ┌──────────────────────────────────┐
                  │ Tier 3: Stable board == expected │
                  │ (excluding unstable_cells)       │
                  └────────────┬─────────────────────┘
                               │
                          Yes  │  No
                      ┌────────┴────────┐
                      v                 v
                  ┌────────┐    ┌───────────────┐
                  │ VALID  │    │ INVALID       │
                  │ Advance│    │ Recompute     │
                  │ Cache  │    │ from current  │
                  └────────┘    └───────────────┘
```

### Handling Unstable Cells

```python
def validate_board(self, actual: BoardState, expected: BoardState) -> bool:
    """
    Compare boards, ignoring cells that are currently unstable.

    Unstable cells are those that fluctuated across recent frames,
    typically caused by user selection highlighting.
    """
    diff = actual.diff(expected)
    meaningful_diff = diff - self._unstable_cells

    return len(meaningful_diff) == 0
```

---

## Implementation Plan

### Phase 1: Data Structures

1. Create `CachedSolution` dataclass
2. Add `expected_boards` computation to `Solution` class
3. Update `SolverStrategy` interface to return full `Solution`

### Phase 2: State Machine Updates

1. Rename `COMPUTING_MOVE` → `COMPUTING_SOLUTION`
2. Add `_cached_solution: Optional[CachedSolution]` to `SolutionManager`
3. Implement cache advancement logic in `VALIDATING_CLEAR`

### Phase 3: Validation Logic

1. Implement Tier 1 fast-path check
2. Integrate existing stability buffer for Tier 2
3. Add `validate_board()` with unstable cell filtering for Tier 3

### Phase 4: Cache Invalidation

1. Detect board divergence from expected
2. Clear cache and trigger recomputation
3. Preserve valid portion of cache when possible (partial invalidation)

---

## API Changes

### SolutionManager

```python
class SolutionManager:
    # New properties
    @property
    def cached_solution(self) -> Optional[CachedSolution]: ...

    @property
    def moves_remaining(self) -> int: ...

    @property
    def total_moves(self) -> int: ...

    # Updated methods
    def update(self, board: BoardState) -> bool:
        """Now manages cached solution lifecycle."""
        ...

    # New methods
    def invalidate_cache(self) -> None:
        """Force cache invalidation and recomputation."""
        ...

    def peek_next_moves(self, count: int = 3) -> List[Move]:
        """Preview upcoming moves without advancing."""
        ...
```

### SolverStrategy

```python
class SolverStrategy(ABC):
    # Existing (unchanged)
    def find_best_move(self, board: BoardState) -> Optional[Move]: ...

    # New method
    def solve_full(self, board: BoardState) -> Solution:
        """Compute complete solution with all moves."""
        ...
```

---

## Edge Cases

### 1. User Executes Wrong Move

```
Scenario: Cache expects cells A,B,C cleared, but user clears D,E,F

Detection: Tier 3 comparison fails (board != expected)
Action: Invalidate cache, recompute from actual board
```

### 2. User Executes Partial Move

```
Scenario: Cache expects 5 cells cleared, user clears 3

Detection: Tier 1 partial match, Tier 3 mismatch
Action: Invalidate cache, recompute from actual board
```

### 3. External Board Change

```
Scenario: Game adds new cells, gravity shifts, etc.

Detection: Board change without expected cell clearing
Action: Invalidate cache, recompute from actual board
```

### 4. Selection Flutter During Display

```
Scenario: User hovers over cells, causing OCR misreads

Detection: Frames differ but stabilize to expected board
Action: Tier 2 filters flutter, cache remains valid
```

### 5. Cache Timeout

```
Scenario: User inactive for extended period

Detection: Cache age > threshold (e.g., 30 seconds)
Action: Invalidate stale cache, recompute fresh solution
```

---

## Performance Considerations

### Computation Cost

| Operation | Current | With Cache |
|-----------|---------|------------|
| First move | ~50-200ms | ~200-500ms (full solution) |
| Subsequent moves | ~50-200ms each | ~0ms (cache lookup) |
| Total (10 moves) | ~500-2000ms | ~200-500ms |

### Memory Usage

```
Per cached solution:
  - Solution object: ~1KB
  - Board states (10 moves): ~10KB
  - Total: ~11KB per cache entry

Single cache = negligible memory impact
```

### Trade-offs

| Aspect | Single-Move | Cached Solution |
|--------|-------------|-----------------|
| Initial latency | Low | Higher |
| Subsequent latency | Consistent | Near-zero |
| Adaptability | High (recomputes) | Lower (uses cache) |
| User feedback | Per-move | Full plan visible |

---

## Future Enhancements

### 1. Move Preview

Display upcoming moves with reduced opacity:
```
Move 1: [BRIGHT GREEN] - Current
Move 2: [FADED GREEN] - Next
Move 3: [VERY FADED] - After that
```

### 2. Solution Branching

When cache invalidates, check if alternate cached solutions exist:
```
Board hash → List[CachedSolution]
```

### 3. Partial Cache Preservation

If user executes moves 1-3 correctly but deviates at move 4:
```
Keep moves 1-3 history
Recompute from move 4 onwards
Append new moves to existing solution
```

### 4. Confidence Scoring

Track cache hit rate to adjust validation strictness:
```
High confidence: Aggressive Tier 1 acceptance
Low confidence: Always wait for Tier 3
```

---

## Testing Strategy

### Unit Tests

1. `CachedSolution` advancement and exhaustion
2. Tier 1/2/3 validation logic
3. Unstable cell filtering
4. Cache invalidation triggers

### Integration Tests

1. Full solve → cache → playback → completion
2. User deviation → cache invalidation → recompute
3. Flutter simulation → stability filtering
4. Timeout → cache expiry

### Manual Testing

1. Normal gameplay with cache enabled
2. Intentional wrong moves
3. Rapid selection (flutter generation)
4. Long idle periods

---

## Rollout Plan

1. **Feature flag:** `use_solution_cache = False` (default off)
2. **Alpha testing:** Enable for debug builds
3. **Metrics collection:** Cache hit rate, invalidation frequency
4. **Gradual rollout:** Enable by default after validation
