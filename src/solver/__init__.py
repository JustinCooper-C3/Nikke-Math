"""
Solver Package - Modular solution framework for Sum to 10 puzzle.

This package provides a pluggable strategy framework for solving the
Sum to 10 grid puzzle. Strategies can be selected at runtime via UI.

Public API:
    - BoardState: Immutable board representation
    - Move: Rectangle move definition
    - Solution: Result of strategy computation
    - SolutionMetrics: Performance statistics
    - SolutionContext: Shared context for strategies
    - SolverStrategy: Abstract base for strategies
    - create_strategy(): Factory function
    - get_strategy_names(): List available strategies
    - get_strategy_info(): Get strategy metadata

Usage:
    from src.solver import create_strategy, BoardState, SolutionContext

    # Create board from OCR result
    board = BoardState.from_ocr(ocr_result)

    # Create context with cancellation support
    context = SolutionContext(board=board)

    # Create and run strategy
    strategy = create_strategy("greedy")
    solution = strategy.solve(context)

    # Access results
    for move in solution.moves:
        print(f"Clear {move.cell_count} cells at ({move.r1},{move.c1})-({move.r2},{move.c2})")
"""

# Core data structures
from .board import BoardState
from .move import Move
from .solution import Solution, SolutionMetrics
from .context import SolutionContext

# Strategy framework
from .base import SolverStrategy
from .factory import (
    create_strategy,
    get_strategy_names,
    get_strategy_info,
    get_default_strategy_name,
    register_strategy,
)

# Import strategies to register them
from . import strategies

__all__ = [
    # Data structures
    "BoardState",
    "Move",
    "Solution",
    "SolutionMetrics",
    "SolutionContext",
    # Strategy framework
    "SolverStrategy",
    "create_strategy",
    "get_strategy_names",
    "get_strategy_info",
    "get_default_strategy_name",
    "register_strategy",
]
