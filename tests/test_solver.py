"""
Test script for solver validation

Uses debug images to test:
1. BoardState creation and hashing
2. Move finding algorithm
3. Solver strategies
4. SolutionManager stability

Usage:
    python test_solver.py
"""

import sys
import time
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import (
    BoardState,
    Move,
    SolutionContext,
    create_strategy,
)
from src.solution_manager import SolutionManager, SolutionState
from src.ocr import create_engine, DEBUG_DIR


def test_board_state():
    """Test BoardState creation and methods."""
    print("\n" + "="*60)
    print("TEST: BoardState")
    print("="*60)

    # Create sample board
    sample_board = [
        [6, 4, 3, 8, 3],
        [8, 6, 7, 7, 6],
        [9, 5, 7, 6, 9],
        [None, None, 3, None, None],
        [8, 3, 7, None, None],
    ]

    board = BoardState.from_ocr(sample_board)
    print(f"  Created board: {board.rows}x{board.cols}")
    print(f"  Cell count: {board.count_cells()}")
    print(f"  Hash: {hash(board)}")

    # Test equality
    board2 = BoardState.from_ocr(sample_board)
    print(f"  Equality test (same data): {board == board2}")
    print(f"  Hash equality: {hash(board) == hash(board2)}")

    # Test diff
    modified = [row[:] for row in sample_board]
    modified[0][0] = None
    board3 = BoardState.from_ocr(modified)
    diff = board.diff(board3)
    print(f"  Diff test (1 cell changed): {diff}")

    print("  [PASS] BoardState tests")
    return True


def test_move_finding():
    """Test finding valid moves that sum to 10."""
    print("\n" + "="*60)
    print("TEST: Move Finding")
    print("="*60)

    # Simple test board with known solutions
    test_board = [
        [1, 9, 3, 7],  # (0,0)+(0,1) = 10, (0,2)+(0,3) = 10
        [5, 5, 2, 8],  # (1,0)+(1,1) = 10, (1,2)+(1,3) = 10
        [4, 6, 1, 9],  # (2,0)+(2,1) = 10, (2,2)+(2,3) = 10
        [3, 7, 8, 2],  # (3,0)+(3,1) = 10, (3,2)+(3,3) = 10
    ]

    board = BoardState.from_ocr(test_board)
    strategy = create_strategy("greedy")
    moves = strategy.find_all_valid_moves(board)

    print(f"  Board size: {board.rows}x{board.cols}")
    print(f"  Found {len(moves)} valid moves")

    # Show first few moves
    for i, move in enumerate(moves[:5]):
        print(f"    Move {i+1}: ({move.r1},{move.c1})->({move.r2},{move.c2}) "
              f"cells={move.cell_count} sum={move.total}")

    if len(moves) > 5:
        print(f"    ... and {len(moves) - 5} more")

    # Verify all moves sum to 10
    all_valid = all(m.total == 10 for m in moves)
    print(f"  All moves sum to 10: {all_valid}")

    if not all_valid:
        print("  [FAIL] Some moves don't sum to 10!")
        return False

    print("  [PASS] Move finding tests")
    return True


def test_solver_strategies():
    """Test solver strategies."""
    print("\n" + "="*60)
    print("TEST: Solver Strategies")
    print("="*60)

    # Test board
    test_board = [
        [1, 9, 3, 7],
        [5, 5, 2, 8],
        [4, 6, 1, 9],
        [3, 7, 8, 2],
    ]

    board = BoardState.from_ocr(test_board)
    initial_cells = board.count_cells()

    print(f"  Initial cells: {initial_cells}")

    # Test greedy strategy
    greedy = create_strategy("greedy")
    context = SolutionContext(board=board)
    solution = greedy.solve(context)

    greedy_cleared = initial_cells - solution.final_board.count_cells()
    print(f"  Greedy: {len(solution.moves)} moves, {greedy_cleared} cells cleared")

    # Test greedy+ strategy (if available)
    try:
        greedy_plus = create_strategy("greedy+")
        context = SolutionContext(board=board)
        solution_plus = greedy_plus.solve(context)
        plus_cleared = initial_cells - solution_plus.final_board.count_cells()
        print(f"  Greedy+: {len(solution_plus.moves)} moves, {plus_cleared} cells cleared")
    except ValueError:
        print(f"  Greedy+: not available")

    # Show solution
    print(f"\n  Solution moves:")
    for i, move in enumerate(solution.moves):
        print(f"    {i+1}. ({move.r1},{move.c1})->({move.r2},{move.c2}) "
              f"clears {move.cell_count} cells")

    print("  [PASS] Solver strategy tests")
    return True


def test_solution_manager():
    """Test SolutionManager single-move-at-a-time state machine."""
    print("\n" + "="*60)
    print("TEST: SolutionManager (Single-Move State Machine)")
    print("="*60)

    manager = SolutionManager(stability_threshold=3, timeout_frames=30)

    test_board = [
        [1, 9, 3, 7],
        [5, 5, 2, 8],
    ]
    board = BoardState.from_ocr(test_board)

    # Test initial state
    print(f"  Initial state: {manager.state}")
    if manager.state != SolutionState.WAITING_STABLE:
        print("  [FAIL] Initial state should be WAITING_STABLE!")
        return False

    # Simulate 3 identical frames to reach stability
    print("  Simulating 3 identical frames...")
    for i in range(3):
        has_move = manager.update(board)
        print(f"    Frame {i+1}: state={manager.state.name}, has_move={has_move}")

    # Check we have a move
    next_move = manager.get_next_move()
    print(f"  State: {manager.state}")
    print(f"  Next move: {next_move}")
    print(f"  Expected cleared cells: {len(manager.expected_cleared)}")

    if manager.state != SolutionState.MOVE_DISPLAYED:
        print("  [FAIL] State should be MOVE_DISPLAYED after 3 stable frames!")
        return False

    if next_move is None:
        print("  [FAIL] Should have a move after stability!")
        return False

    # Simulate executing the move - clear the expected cells
    print("\n  Simulating move execution (clearing cells)...")
    modified_board_data = [row[:] for row in test_board]
    for r, c in next_move.cells:
        modified_board_data[r][c] = None
    modified_board = BoardState.from_ocr(modified_board_data)

    # Feed the modified board
    for i in range(3):
        has_move = manager.update(modified_board)
        print(f"    Frame {i+1}: state={manager.state.name}, has_move={has_move}")

    # Check state after validation
    print(f"  State after clear: {manager.state}")
    new_move = manager.get_next_move()
    print(f"  Next move after clear: {new_move}")

    # Test reset
    manager.reset()
    print(f"  After reset - state: {manager.state}")

    if manager.state != SolutionState.WAITING_STABLE:
        print("  [FAIL] State after reset should be WAITING_STABLE!")
        return False

    print("  [PASS] SolutionManager tests")
    return True


def test_with_debug_image():
    """Test solver with actual debug image from OCR."""
    print("\n" + "="*60)
    print("TEST: Debug Image Integration")
    print("="*60)

    debug_images = list(DEBUG_DIR.glob("debug_*.png"))

    if not debug_images:
        print("  [SKIP] No debug images found")
        return True

    # Use most recent debug image
    debug_image = sorted(debug_images)[-1]
    print(f"  Loading: {debug_image.name}")

    img = Image.open(debug_image)
    print(f"  Image size: {img.size}")

    # Create OCR engine and process
    engine = create_engine("template")
    ocr_result = engine.process(img)

    if ocr_result.grid_info is None:
        print("  [SKIP] Grid not detected in image")
        return True

    print(f"  Grid detected: {ocr_result.grid_info.rows}x{ocr_result.grid_info.cols}")
    print(f"  OCR confidence: {ocr_result.confidence*100:.1f}%")
    print(f"  Total cells: {ocr_result.total_cells}")
    print(f"  Uncertain: {ocr_result.uncertain_count}")

    # Create board state
    board = BoardState.from_ocr(ocr_result.board)
    print(f"  BoardState cells: {board.count_cells()}")

    # Find moves
    strategy = create_strategy("greedy")
    moves = strategy.find_all_valid_moves(board)
    print(f"  Valid moves found: {len(moves)}")

    if moves:
        # Run solver
        context = SolutionContext(board=board)
        solution = strategy.solve(context)
        cleared = board.count_cells() - solution.final_board.count_cells()
        print(f"  Solution: {len(solution.moves)} moves, {cleared} cells clearable")

        # Show first 3 moves
        print(f"\n  First 3 moves:")
        for i, move in enumerate(solution.moves[:3]):
            print(f"    {i+1}. ({move.r1},{move.c1})->({move.r2},{move.c2}) "
                  f"clears {move.cell_count} cells")

    print("  [PASS] Debug image integration tests")
    return True


def test_realistic_board():
    """Test solver with realistic board data (from debug image screenshot)."""
    print("\n" + "="*60)
    print("TEST: Realistic Board (Hardcoded from Screenshot)")
    print("="*60)

    # Board extracted manually from debug_20251211_124609_669.png
    # 16 rows x 10 cols, reading left to right, top to bottom
    realistic_board = [
        [6, 4, 3, 8, 3, None, None, None, None, None],  # Row 0 (partial)
        [8, 6, 7, 7, 6, 9, 7, 1, None, None],           # Row 1
        [9, 5, 7, 6, 9, 9, 5, 5, None, None],           # Row 2
        [None, None, 3, None, None, 6, 8, 3, 8, None],  # Row 3
        [8, 3, 7, None, None, 2, 7, 1, 4, 4],           # Row 4
        [8, 1, 2, 2, None, 9, 7, 9, 6, 3],              # Row 5
        [7, 8, 1, 2, 5, 6, 2, 2, 7, 9],                 # Row 6
        [8, 4, 7, 7, 6, 6, 2, 3, 2, 1],                 # Row 7
        [7, 2, 6, 4, 3, 8, 1, 8, 8, 3],                 # Row 8
        [6, 5, 7, 6, 7, 7, 2, 3, 3, 1],                 # Row 9
        [4, 1, 5, 1, 4, 9, 4, 4, 8, 2],                 # Row 10
        [4, 2, 7, 3, 4, 4, 5, 8, 9, 3],                 # Row 11
        [8, 1, 2, 2, 2, 1, 1, 2, 3, 9],                 # Row 12
        [6, 8, 2, 8, 2, 6, 1, 5, 8, 9],                 # Row 13
        [4, 1, 9, 2, 9, 3, 8, 9, 6, 6],                 # Row 14
        [4, 6, 8, 3, 1, 5, 5, 9, 3, 5],                 # Row 15
    ]

    board = BoardState.from_ocr(realistic_board)
    initial_cells = board.count_cells()

    print(f"  Board size: {board.rows}x{board.cols}")
    print(f"  Initial cells: {initial_cells}")

    # Find all valid moves
    strategy = create_strategy("greedy")
    moves = strategy.find_all_valid_moves(board)
    print(f"  Valid moves found: {len(moves)}")

    if len(moves) == 0:
        print("  [FAIL] No valid moves found on realistic board!")
        return False

    # Show some example moves
    print(f"\n  Sample valid moves (first 5):")
    for i, move in enumerate(moves[:5]):
        # Get the cell values
        values = [board.get_cell(r, c) for r, c in move.cells]
        values_str = "+".join(str(v) for v in values)
        print(f"    {i+1}. ({move.r1},{move.c1})->({move.r2},{move.c2}): "
              f"{values_str}={move.total}, clears {move.cell_count} cells")

    # Run solver
    print(f"\n  Running greedy solver...")
    start = time.perf_counter()
    context = SolutionContext(board=board)
    solution = strategy.solve(context)
    elapsed = (time.perf_counter() - start) * 1000

    cleared = initial_cells - solution.final_board.count_cells()
    remaining = solution.final_board.count_cells()

    print(f"  Solve time: {elapsed:.1f}ms")
    print(f"  Solution: {len(solution.moves)} moves")
    print(f"  Cells cleared: {cleared}/{initial_cells}")
    print(f"  Cells remaining: {remaining}")

    # Show first 5 moves of solution
    print(f"\n  Solution (first 5 moves):")
    temp_board = board
    for i, move in enumerate(solution.moves[:5]):
        values = [temp_board.get_cell(r, c) for r, c in move.cells]
        values_str = "+".join(str(v) for v in values if v is not None)
        print(f"    {i+1}. ({move.r1},{move.c1})->({move.r2},{move.c2}): "
              f"{values_str}={move.total}, clears {move.cell_count}")
        temp_board = temp_board.apply_move(move)

    if len(solution.moves) > 5:
        print(f"    ... and {len(solution.moves) - 5} more moves")

    # Validate solution
    if cleared == 0:
        print("  [FAIL] No cells cleared!")
        return False

    print(f"\n  [PASS] Realistic board test - {cleared} cells clearable")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# SOLVER VALIDATION TESTS")
    print("#"*60)

    results = []

    results.append(("BoardState", test_board_state()))
    results.append(("Move Finding", test_move_finding()))
    results.append(("Solver Strategies", test_solver_strategies()))
    results.append(("SolutionManager", test_solution_manager()))
    results.append(("Debug Image", test_with_debug_image()))
    results.append(("Realistic Board", test_realistic_board()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
