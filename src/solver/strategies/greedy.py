"""
Greedy Strategy - Always picks the move clearing the most cells.
"""

import time
from typing import List

from ..base import SolverStrategy
from ..board import BoardState
from ..move import Move
from ..context import SolutionContext
from ..solution import Solution, SolutionMetrics
from ..factory import register_strategy


@register_strategy
class GreedyStrategy(SolverStrategy):
    """
    Greedy strategy that always picks the move clearing the most cells.

    This is the fastest strategy - essentially instant for any board size.
    It doesn't consider future moves, just maximizes cells cleared at each step.
    """
    name = "greedy"
    description = "Greedy (instant) - Always picks move clearing most cells"
    timeout_sec = 1.0  # Effectively instant

    def solve(self, context: SolutionContext) -> Solution:
        """
        Compute greedy solution - always pick move with most cells.

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

            # Pick move with most cells
            best_move = max(valid_moves, key=lambda m: m.cell_count)
            moves.append(best_move)
            total_cleared += best_move.cell_count

            board = board.apply_move(best_move)
            board_states.append(board)

            # Report progress (estimate based on cells remaining)
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
