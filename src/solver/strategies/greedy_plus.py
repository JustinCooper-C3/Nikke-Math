"""
Greedy Plus Strategy - Greedy with tiebreaker for future move potential.

When multiple moves have the same cell count, picks the one that
enables the most future moves on the resulting board.
"""

import logging
import time
from typing import List, Tuple

logger = logging.getLogger(__name__)

from ..base import SolverStrategy
from ..board import BoardState
from ..move import Move
from ..context import SolutionContext
from ..solution import Solution, SolutionMetrics
from ..factory import register_strategy


@register_strategy
class GreedyPlusStrategy(SolverStrategy):
    """
    Greedy strategy with intelligent tiebreaking.

    Like basic greedy, always picks moves clearing the most cells.
    When multiple moves tie for cell count, uses tiebreakers:
      1. Primary: Most future moves enabled (lookahead 1)
      2. Secondary: Smallest rectangle area (more compact)

    Slightly slower than basic greedy due to tiebreaker evaluation,
    but typically finds better solutions.
    """
    name = "greedy_plus"
    description = "Greedy Plus (smart) - Tiebreaks by future move potential"
    timeout_sec = 5.0  # Allow more time for tiebreaker evaluation

    def solve(self, context: SolutionContext) -> Solution:
        """
        Compute greedy solution with tiebreaking.

        Args:
            context: Solution context with board and cancellation

        Returns:
            Solution with moves and metrics
        """
        start_time = time.perf_counter()

        board = context.board
        moves: List[Move] = []
        board_states: List[BoardState] = [board]
        states_explored = 0
        total_cleared = 0

        while True:
            if self._check_cancelled(context):
                return self._build_solution(
                    moves, board_states, total_cleared, states_explored,
                    start_time, was_cancelled=True
                )

            valid_moves = self.find_all_valid_moves(board)
            states_explored += len(valid_moves)

            if not valid_moves:
                break

            # Find best move with tiebreaking
            best_move, extra_explored = self._select_best_move(
                board, valid_moves, context
            )
            states_explored += extra_explored

            moves.append(best_move)
            total_cleared += best_move.cell_count

            board = board.apply_move(best_move)
            board_states.append(board)

            # Report progress
            initial_cells = context.board.count_cells()
            if initial_cells > 0:
                progress = total_cleared / initial_cells
                context.report_progress(
                    min(0.99, progress),
                    f"{len(moves)} moves, {total_cleared} cells cleared"
                )

        return self._build_solution(
            moves, board_states, total_cleared, states_explored,
            start_time, was_cancelled=False
        )

    def find_best_move(self, board: BoardState) -> Move | None:
        """
        Find the single best move with tiebreaker logic.

        Overrides base class to apply tiebreaking when multiple
        moves have the same cell count.

        Args:
            board: Current board state

        Returns:
            Best Move object, or None if no valid moves
        """
        valid_moves = self.find_all_valid_moves(board)

        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        # Find max cell count
        max_cells = max(m.cell_count for m in valid_moves)

        # Filter to moves with max cell count (ties)
        tied_moves = [m for m in valid_moves if m.cell_count == max_cells]

        if len(tied_moves) == 1:
            return tied_moves[0]

        # Tiebreaker: simulate greedy completion for each tied move
        # Pick the one that leads to most total cells cleared
        logger.info(f"[GreedyPlus] Tiebreaker: {len(tied_moves)} moves with {max_cells} cells, simulating completions...")
        scored_moves: List[Tuple[Move, int, int]] = []  # (move, total_clearable, area)

        for move in tied_moves:
            # Simulate greedy completion from this move
            total_clearable = move.cell_count + self._simulate_greedy_completion(
                board.apply_move(move)
            )
            scored_moves.append((move, total_clearable, move.area))

        # Sort by: most cells clearable (desc), then smallest area (asc)
        scored_moves.sort(key=lambda x: (-x[1], x[2]))

        best = scored_moves[0]
        logger.info(f"[GreedyPlus] Selected move leading to {best[1]} total cells clearable (area={best[2]})")

        return best[0]

    def _simulate_greedy_completion(self, board: BoardState) -> int:
        """
        Simulate greedy completion and return total cells clearable.

        Runs a fast greedy simulation (no tiebreaking) to estimate
        how many cells can be cleared from this board state.

        Args:
            board: Board state to simulate from

        Returns:
            Total cells that can be cleared with greedy play
        """
        total_cleared = 0

        while True:
            moves = self.find_all_valid_moves(board)
            if not moves:
                break

            # Simple greedy: pick move with most cells
            best_move = max(moves, key=lambda m: m.cell_count)
            total_cleared += best_move.cell_count
            board = board.apply_move(best_move)

        return total_cleared

    def _select_best_move(
        self,
        board: BoardState,
        valid_moves: List[Move],
        context: SolutionContext
    ) -> Tuple[Move, int]:
        """
        Select best move using tiebreaker logic.

        Args:
            board: Current board state
            valid_moves: List of valid moves to choose from
            context: Solution context for cancellation check

        Returns:
            Tuple of (best_move, states_explored_in_tiebreak)
        """
        if len(valid_moves) == 1:
            return valid_moves[0], 0

        # Find max cell count
        max_cells = max(m.cell_count for m in valid_moves)

        # Filter to moves with max cell count (ties)
        tied_moves = [m for m in valid_moves if m.cell_count == max_cells]

        if len(tied_moves) == 1:
            return tied_moves[0], 0

        # Tiebreaker: simulate greedy completion for each tied move
        states_explored = 0
        scored_moves: List[Tuple[Move, int, int]] = []  # (move, total_clearable, area)

        for move in tied_moves:
            if self._check_cancelled(context):
                # If cancelled, just return first tied move
                return tied_moves[0], states_explored

            # Simulate greedy completion from this move
            resulting_board = board.apply_move(move)
            total_clearable = move.cell_count + self._simulate_greedy_completion(resulting_board)
            scored_moves.append((move, total_clearable, move.area))
            states_explored += 1  # Count simulation as one exploration

        # Sort by: most cells clearable (desc), then smallest area (asc)
        scored_moves.sort(key=lambda x: (-x[1], x[2]))

        return scored_moves[0][0], states_explored

    def _build_solution(
        self,
        moves: List[Move],
        board_states: List[BoardState],
        total_cleared: int,
        states_explored: int,
        start_time: float,
        was_cancelled: bool
    ) -> Solution:
        """Build Solution object from computation results."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return Solution(
            moves=moves,
            total_cleared=total_cleared,
            is_complete=not was_cancelled and len(moves) > 0,
            was_cancelled=was_cancelled,
            board_states=board_states,
            metrics=SolutionMetrics(
                computation_time_ms=elapsed_ms,
                states_explored=states_explored,
                pruned_branches=0,
                strategy_name=self.name
            )
        )
