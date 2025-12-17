# Solution Cache Improvements Plan

## Final Design (Simplified)

Based on user feedback:
- Delayed wrong-move detection is acceptable
- Only verify expected cells are zero, then advance
- Require 3 stable frames before advancing
- Immediately recompute when cache exhausts

### New State Flow

```
MOVE_DISPLAYED
    │
    └── Expected cells all zero?
            │
            ├── NO → Stay (ignore all other board changes)
            │
            └── YES → VALIDATING_CLEAR
                        │
                        └── 3 consecutive frames with expected cells = 0?
                                │
                                ├── NO → Stay in VALIDATING_CLEAR
                                │
                                └── YES → Advance to next move
                                            │
                                            ├── More moves? → MOVE_DISPLAYED
                                            │
                                            └── Cache exhausted? → Recompute immediately

```

### Key Simplifications

| Before | After |
|--------|-------|
| Full board comparison in MOVE_DISPLAYED | Only check expected cells |
| Tier 1/2/3 validation | Single check: expected cells = 0 for 3 frames |
| Complex unstable cell tracking | Not needed |
| State oscillation possible | Impossible - only advance when cells cleared |
| Cache invalidation on mismatch | Never invalidate - trust cache until exhausted |

### Implementation Changes

1. **`_handle_move_displayed()`**: Replace `board != last_stable_board` with `_expected_cells_are_zero(board)`
2. **`_handle_validating_clear()`**: Simplify to just stability check on expected cells
3. **Remove**: Tier 3 board comparison, unstable cell filtering, cache invalidation logic
4. **`_display_current_move()`**: When exhausted, trigger immediate recompute instead of WAITING_STABLE

---

## Original Analysis (Reference)

## Log Analysis Summary

Analyzed `solver.log` from session 15:09:01 - 15:12:48.

### Observed Problems

**Problem 1: False Positive Board Changes During Selection**
```
15:09:18 [INFO] Displaying move 2/2: clearing 3 cells
15:09:18 [INFO] State[MOVE_DISPLAYED]: board changed, transitioning to VALIDATING_CLEAR
15:09:18 [DEBUG] State[VALIDATING_CLEAR]: Tier 2 - still stabilizing, 190 unstable cells
15:09:19 [DEBUG] State[VALIDATING_CLEAR]: no change detected, returning to MOVE_DISPLAYED
15:09:19 [INFO] State[MOVE_DISPLAYED]: board changed, transitioning to VALIDATING_CLEAR
... (repeats 3-4 times before actual clear)
```

**Root Cause:** The comparison `board != self._last_stable_board` in `_handle_move_displayed()` (line 294) uses exact equality. When user is selecting cells:
- Selection highlighting changes cell appearance
- OCR may misread highlighted cells (color shift)
- Even 1 cell differing triggers VALIDATING_CLEAR

**Impact:** Unnecessary state thrashing between MOVE_DISPLAYED and VALIDATING_CLEAR.

---

**Problem 2: Unstable Cells Cleared Too Early in Tier 3**
```python
# Line 342 in _handle_validating_clear:
self._unstable_cells = set()  # <-- Cleared here

# Line 346: Then Tier 3 tries to use it
if self._cached_solution.validate_board_match(first_board, self._unstable_cells):
    # self._unstable_cells is now empty!
```

**Root Cause:** `_unstable_cells` is reset to empty *before* being passed to `validate_board_match()`, making the unstable cell filtering useless.

**Impact:** Tier 3 can't filter out cells that were unstable, causing false negatives.

---

**Problem 3: No Hysteresis in Board Change Detection**
```
15:09:19 [DEBUG] State[VALIDATING_CLEAR]: no change detected, returning to MOVE_DISPLAYED
15:09:19 [INFO] State[MOVE_DISPLAYED]: board changed, transitioning to VALIDATING_CLEAR
```

**Root Cause:** Immediate re-entry into VALIDATING_CLEAR on very next frame after returning to MOVE_DISPLAYED. No cooldown or confidence threshold.

**Impact:** Rapid state oscillation wastes CPU and creates visual flicker.

---

**Problem 4: Recomputation After Cache Exhausted**
```
15:09:20 [INFO] Cache exhausted, no more moves
15:09:21 [INFO] State[WAITING_STABLE]: board stable, transitioning to COMPUTING_SOLUTION
15:09:26 [INFO] State[COMPUTING_SOLUTION]: computed 1 moves, 3 total cells (5100.6ms)
```

**Root Cause:** When cache exhausts, system returns to WAITING_STABLE and recomputes even though the board may still have valid moves from the game's perspective.

**Impact:** 5+ second recomputation delay between "rounds" of the solution.

---

## Improvement Options

### Option A: Tolerant Board Comparison

**Change:** Replace exact equality with threshold-based comparison.

```python
def _boards_significantly_different(self, a: BoardState, b: BoardState, threshold: int = 3) -> bool:
    """
    Check if boards differ by more than threshold cells.

    Ignores minor OCR noise while detecting real changes.
    """
    diff_count = len(a.diff(b))
    return diff_count > threshold
```

**Pros:**
- Simple to implement
- Filters out 1-2 cell OCR noise

**Cons:**
- May delay detection of small legitimate changes
- Threshold tuning required

---

### Option B: Expected Cells Focus Mode

**Change:** In MOVE_DISPLAYED state, only monitor the cells in `_expected_cleared` for changes, ignore all other cells.

```python
def _expected_cells_changed(self, board: BoardState) -> bool:
    """Only check if expected-to-clear cells have changed."""
    for r, c in self._expected_cleared:
        current = board.get_cell(r, c)
        previous = self._last_stable_board.get_cell(r, c)
        if current != previous:
            return True
    return False
```

**Pros:**
- Precisely targets what matters
- Immune to noise elsewhere on board

**Cons:**
- Won't detect if user clears wrong cells (delayed detection)

---

### Option C: Validation Cooldown / Hysteresis

**Change:** After returning from VALIDATING_CLEAR to MOVE_DISPLAYED, require N frames before re-entering validation.

```python
# In __init__:
self._validation_cooldown: int = 0

# In _handle_move_displayed:
if self._validation_cooldown > 0:
    self._validation_cooldown -= 1
    return True  # Don't check for changes yet

# After returning from validation:
self._validation_cooldown = 3  # Skip next 3 frames
```

**Pros:**
- Prevents rapid oscillation
- Simple state machine addition

**Cons:**
- Adds latency to valid detections
- Another magic number to tune

---

### Option D: Cumulative Unstable Cell Tracking

**Change:** Don't clear `_unstable_cells` between validation attempts. Accumulate over time and only clear on successful validation or full reset.

```python
# In _handle_validating_clear, REMOVE:
# self._unstable_cells = set()  # <-- Delete this line

# Instead, preserve unstable cells and only clear on success:
def _advance_to_next_move(self, new_board: BoardState) -> bool:
    self._unstable_cells = set()  # Clear only on success
    ...
```

**Pros:**
- Builds up knowledge of noisy cells over time
- Better filtering in Tier 3

**Cons:**
- May accumulate stale data if cells stabilize

---

### Option E: Selection Detection Heuristic

**Change:** Detect when user is actively selecting (many cells changing to similar values) and pause validation.

```python
def _is_selection_in_progress(self, board: BoardState) -> bool:
    """
    Detect selection highlighting pattern.

    When user drags selection, many cells may shift to similar
    brightness/color, causing OCR to read them as 0 or wrong values.
    """
    diff = set(self._last_stable_board.diff(board))
    if len(diff) > 10:  # Many cells changed
        # Check if changes look like selection (values going to 0)
        zeroed = sum(1 for r, c in diff if board.get_cell(r, c) == 0)
        if zeroed > len(diff) * 0.5:
            return True
    return False
```

**Pros:**
- Directly addresses root cause
- Can pause validation during selection

**Cons:**
- Heuristic may have false positives/negatives
- Depends on OCR behavior during selection

---

### Option F: Two-Phase Validation with Confirmation

**Change:** Require board change to persist for N frames before entering VALIDATING_CLEAR.

```python
# Track consecutive "different" frames
self._change_confirmation_count: int = 0
CHANGE_CONFIRMATION_THRESHOLD = 2

def _handle_move_displayed(self, board: BoardState) -> bool:
    if board != self._last_stable_board:
        self._change_confirmation_count += 1
        if self._change_confirmation_count >= CHANGE_CONFIRMATION_THRESHOLD:
            # Confirmed change, enter validation
            self._change_confirmation_count = 0
            self._state = SolutionState.VALIDATING_CLEAR
            ...
    else:
        self._change_confirmation_count = 0  # Reset on stable frame
    return True
```

**Pros:**
- Filters transient changes
- No heuristics about selection patterns

**Cons:**
- Adds ~200ms latency (2 frames at 10 FPS)

---

## Recommended Approach: Hybrid

Combine multiple options for robustness:

1. **Option D (Cumulative Unstable Cells)** - Fix the bug where unstable cells are cleared too early
2. **Option F (Two-Phase Confirmation)** - Require 2 consecutive different frames before entering validation
3. **Option B (Expected Cells Focus)** - As a Tier 0 fast-path before full board comparison

### Proposed State Flow

```
MOVE_DISPLAYED
    │
    ├── Frame differs?
    │       │
    │       └── NO → Reset confirmation counter, stay
    │       │
    │       └── YES → Increment confirmation counter
    │                   │
    │                   ├── Counter < 2 → Stay (wait for confirmation)
    │                   │
    │                   └── Counter >= 2 → Check expected cells
    │                                        │
    │                                        ├── Expected cells cleared? → Tier 1 PASS
    │                                        │
    │                                        └── Not cleared → VALIDATING_CLEAR
    │
VALIDATING_CLEAR
    │
    ├── Tier 1: Expected cells all empty? → PASS → Advance
    │
    ├── Tier 2: Wait for stability (3 frames)
    │       │
    │       └── Accumulate unstable cells (don't clear!)
    │
    └── Tier 3: Board matches expected (filtering unstable cells)?
            │
            ├── YES → Advance, clear unstable cells
            │
            └── NO → Analyze what happened
                    │
                    ├── Expected cells cleared + extra changes → Invalidate
                    │
                    ├── Partial clear → Wait more / Invalidate
                    │
                    └── No change → Return to MOVE_DISPLAYED (with cooldown)
```

---

## Implementation Priority

| Priority | Option | Effort | Impact |
|----------|--------|--------|--------|
| 1 | D - Fix unstable cells bug | Low | High |
| 2 | F - Change confirmation | Low | High |
| 3 | B - Expected cells focus | Medium | Medium |
| 4 | C - Validation cooldown | Low | Medium |
| 5 | A - Tolerant comparison | Low | Low |

---

## Questions for User

1. How often does the user select incorrect cells (wrong move)? This affects how aggressive we can be with filtering.

2. Is there a visual indicator in the game when selection is in progress? (e.g., highlight color)

3. What is acceptable latency between user completing a move and the next move being displayed?

4. Should the system ever invalidate cache, or should it always trust the cached solution until exhausted?
