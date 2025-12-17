"""
Beam Search Strategy - Bounded lookahead optimization for highest score.

Uses beam search to explore multiple move sequences up to a fixed depth,
keeping only the top beam_width candidates at each level. Balances solution
quality with predictable performance.

Replaces greedy_plus which suffered from unbounded simulation time.
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..base import SolverStrategy
from ..board import BoardState
from ..move import Move
from ..context import SolutionContext
from ..solution import Solution, SolutionMetrics
from ..factory import register_strategy

logger = logging.getLogger(__name__)


@dataclass
class BeamNode:
    """
    Node in beam search tree.

    Represents a board state reachable via a sequence of moves,
    with scoring for beam pruning.

    Attributes:
        board: Current board state
        path: Moves taken to reach this state (as tuple for immutability)
        cleared: Total cells cleared on path
        score: cleared + heuristic estimate (for sorting)
    """
    board: BoardState
    path: Tuple[Move, ...]
    cleared: int
    score: float = 0.0

    def __lt__(self, other: "BeamNode") -> bool:
        """Max-heap ordering: higher score = higher priority."""
        return self.score > other.score

    @property
    def first_move(self) -> Optional[Move]:
        """Get the first move in path (the one to execute now)."""
        return self.path[0] if self.path else None

    @property
    def depth(self) -> int:
        """Current depth in search tree."""
        return len(self.path)


@register_strategy
class BeamSearchStrategy(SolverStrategy):
    """
    Bounded beam search solver with configurable lookahead.

    Explores multiple move sequences up to a fixed depth,
    keeping only the top beam_width candidates at each level.
    Balances solution quality with predictable performance.

    Algorithm:
        1. Start with current board as root node
        2. For each depth level:
           - Expand all nodes in beam by trying all valid moves
           - Score each resulting state
           - Keep only top beam_width nodes
        3. Return first move of highest-scoring path
        4. Repeat for full solution

    Parameters:
        depth: Number of moves to look ahead (default 4)
        beam_width: Candidates to keep per level (default 8)
        heuristic_weight: Weight for remaining cell estimate (default 0.3)

    Performance:
        - ~160ms per move selection (vs 5s+ for greedy_plus)
        - ~3 seconds for full 20-move solution
        - Finds better solutions than greedy by considering cascades
    """
    name = "beam"
    description = "Beam Search (balanced) - Bounded lookahead optimization"
    timeout_sec = 10.0

    def __init__(self, depth: int = 2, beam_width: int = 5,
                 heuristic_weight: float = 0.3):
        """
        Initialize beam search strategy.

        Args:
            depth: Moves to look ahead (2-4 recommended, higher = slower)
            beam_width: Candidates kept per level (4-8 recommended)
            heuristic_weight: Weight for remaining cell heuristic (0.0-0.5)

        Default parameters (depth=2, beam_width=5) provide:
            - ~4 seconds for full 160-cell board solution
            - +2-5% more cells cleared vs greedy
            - Good balance of quality and speed
        """
        self.depth = depth
        self.beam_width = beam_width
        self.heuristic_weight = heuristic_weight

    def solve(self, context: SolutionContext) -> Solution:
        """
        Compute full solution using beam search at each step.

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
        initial_cells = board.count_cells()

        while True:
            if self._check_cancelled(context):
                return self._build_solution(
                    moves, board_states, total_cleared, states_explored,
                    start_time, was_cancelled=True
                )

            best_move, explored = self._select_best_move(board, context)
            states_explored += explored

            if best_move is None:
                break

            moves.append(best_move)
            total_cleared += best_move.cell_count
            board = board.apply_move(best_move)
            board_states.append(board)

            # Report progress
            if initial_cells > 0:
                progress = min(0.99, total_cleared / initial_cells)
                context.report_progress(
                    progress,
                    f"{len(moves)} moves, {total_cleared} cells cleared"
                )

            logger.debug(
                f"[BeamSearch] Move {len(moves)}: clearing {best_move.cell_count} cells, "
                f"total {total_cleared}/{initial_cells}"
            )

        logger.info(
            f"[BeamSearch] Solution complete: {len(moves)} moves, "
            f"{total_cleared} cells, {states_explored} states explored"
        )

        return self._build_solution(
            moves, board_states, total_cleared, states_explored,
            start_time, was_cancelled=False
        )

    def _select_best_move(
        self,
        board: BoardState,
        context: SolutionContext
    ) -> Tuple[Optional[Move], int]:
        """
        Use beam search to find best first move.

        Args:
            board: Current board state
            context: Solution context for cancellation

        Returns:
            Tuple of (best_move, states_explored)
        """
        # Initialize beam with root node
        root = BeamNode(board=board, path=(), cleared=0)
        root.score = self._calculate_score(root)
        beam = [root]
        states_explored = 1

        # Expand through depth levels
        for d in range(self.depth):
            if self._check_cancelled(context):
                break

            beam, explored = self._expand_beam(beam, context)
            states_explored += explored

            if not beam:
                break

        # Return first move of best path
        if beam and beam[0].path:
            return beam[0].path[0], states_explored

        # Fallback: simple greedy if beam search found nothing
        valid_moves = self.find_all_valid_moves(board)
        states_explored += len(valid_moves)

        if valid_moves:
            best = max(valid_moves, key=lambda m: m.cell_count)
            return best, states_explored

        return None, states_explored

    def _expand_beam(
        self,
        beam: List[BeamNode],
        context: SolutionContext
    ) -> Tuple[List[BeamNode], int]:
        """
        Expand all nodes in beam by one level.

        Args:
            beam: Current beam nodes
            context: Solution context for cancellation

        Returns:
            Tuple of (new_beam, states_explored)
        """
        candidates: List[BeamNode] = []
        states_explored = 0

        for node in beam:
            if self._check_cancelled(context):
                break

            valid_moves = self.find_all_valid_moves(node.board)
            states_explored += len(valid_moves)

            # Sort moves by cell count (better moves first for pruning)
            valid_moves.sort(key=lambda m: m.cell_count, reverse=True)

            for move in valid_moves:
                new_board = node.board.apply_move(move)
                new_node = BeamNode(
                    board=new_board,
                    path=node.path + (move,),
                    cleared=node.cleared + move.cell_count
                )
                new_node.score = self._calculate_score(new_node)
                candidates.append(new_node)

                # Early termination: if board is empty, we found optimal path
                if new_board.count_cells() == 0:
                    return [new_node], states_explored

        # Keep top beam_width nodes
        candidates.sort()
        return candidates[:self.beam_width], states_explored

    def _calculate_score(self, node: BeamNode) -> float:
        """
        Calculate node score for beam pruning.

        Score = cells cleared so far + heuristic estimate of remaining potential.
        Higher score = more promising path.

        Args:
            node: Beam node to score

        Returns:
            Score value (higher is better)
        """
        path_score = node.cleared

        # Heuristic: estimate remaining clearable cells
        # Assume some fraction of remaining cells can be cleared
        remaining_cells = node.board.count_cells()
        heuristic = remaining_cells * self.heuristic_weight

        return path_score + heuristic

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
