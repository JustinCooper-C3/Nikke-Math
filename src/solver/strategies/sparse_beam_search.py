"""
Sparse Beam Search Strategy - Optimized for sparse grids with scattered cells.

Extends beam search with adaptive parameters, sparse-aware heuristics,
cluster-based prioritization, and spatial indexing for efficient move finding.

Designed to handle grids where many cells are empty (0 or None), which cause
issues for the standard beam search due to:
- Fewer valid moves making beam width less effective
- Simple heuristic failing to differentiate paths
- O(n^4) rectangle search wasted on empty regions
"""

import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict

from ..base import SolverStrategy
from ..board import BoardState
from ..move import Move
from ..context import SolutionContext
from ..solution import Solution, SolutionMetrics
from ..factory import register_strategy

logger = logging.getLogger(__name__)


@dataclass
class CellCluster:
    """
    A group of connected non-empty cells.

    Attributes:
        cells: Set of (row, col) positions in this cluster
        bounds: Bounding box as (min_row, min_col, max_row, max_col)
        total_value: Sum of all cell values in cluster
    """
    cells: Set[Tuple[int, int]]
    bounds: Tuple[int, int, int, int]
    total_value: int

    @property
    def size(self) -> int:
        return len(self.cells)

    def contains(self, row: int, col: int) -> bool:
        return (row, col) in self.cells

    def intersects_rect(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if cluster bounds intersect with rectangle."""
        min_r, min_c, max_r, max_c = self.bounds
        return not (r2 < min_r or r1 > max_r or c2 < min_c or c1 > max_c)


@dataclass
class SparseBeamNode:
    """
    Node in sparse beam search tree with enhanced scoring.

    Attributes:
        board: Current board state
        path: Moves taken to reach this state
        cleared: Total cells cleared on path
        clusters_cleared: Number of complete clusters cleared
        connectivity_score: Bonus for improving cell connectivity
        score: Combined score for beam pruning
    """
    board: BoardState
    path: Tuple[Move, ...]
    cleared: int
    clusters_cleared: int = 0
    connectivity_score: float = 0.0
    score: float = 0.0

    def __lt__(self, other: "SparseBeamNode") -> bool:
        """Max-heap ordering: higher score = higher priority."""
        return self.score > other.score

    @property
    def first_move(self) -> Optional[Move]:
        return self.path[0] if self.path else None

    @property
    def depth(self) -> int:
        return len(self.path)


class SpatialIndex:
    """
    Spatial index for efficient move finding on sparse grids.

    Pre-computes cell locations and cluster bounds to prune
    rectangle enumeration to relevant regions only.

    Clusters are computed lazily on first access to reduce startup cost.
    """
    # Class-level cache to avoid redundant builds for same board
    _cache: Dict[int, "SpatialIndex"] = {}

    def __init__(
        self,
        cell_positions: Set[Tuple[int, int]],
        row_has_cells: Set[int],
        col_has_cells: Set[int],
        density: float,
        board: BoardState
    ):
        self.cell_positions = cell_positions
        self.row_has_cells = row_has_cells
        self.col_has_cells = col_has_cells
        self.density = density
        self._board = board
        self._clusters: Optional[List[CellCluster]] = None  # Lazy

    @property
    def clusters(self) -> List[CellCluster]:
        """Lazily compute clusters on first access."""
        if self._clusters is None:
            self._clusters = self._find_clusters(self._board, self.cell_positions)
        return self._clusters

    @classmethod
    def build(cls, board: BoardState) -> "SpatialIndex":
        """Build spatial index from board state (cached, lazy clusters)."""
        # Check cache first
        board_hash = hash(board)
        if board_hash in cls._cache:
            return cls._cache[board_hash]

        cell_positions: Set[Tuple[int, int]] = set()
        row_has_cells: Set[int] = set()
        col_has_cells: Set[int] = set()

        total_cells = board.rows * board.cols

        for r in range(board.rows):
            for c in range(board.cols):
                val = board.get_cell(r, c)
                if val is not None and val != 0:
                    cell_positions.add((r, c))
                    row_has_cells.add(r)
                    col_has_cells.add(c)

        density = len(cell_positions) / total_cells if total_cells > 0 else 0.0

        # Clusters computed lazily on first access
        result = cls(
            cell_positions=cell_positions,
            row_has_cells=row_has_cells,
            col_has_cells=col_has_cells,
            density=density,
            board=board
        )

        # Cache result (limit cache size to prevent memory issues)
        if len(cls._cache) > 100:
            cls._cache.clear()
        cls._cache[board_hash] = result

        return result

    @classmethod
    def clear_cache(cls):
        """Clear the spatial index cache."""
        cls._cache.clear()

    @staticmethod
    def _find_clusters(
        board: BoardState,
        cell_positions: Set[Tuple[int, int]]
    ) -> List[CellCluster]:
        """Find connected clusters using flood-fill."""
        visited: Set[Tuple[int, int]] = set()
        clusters: List[CellCluster] = []

        for start_pos in cell_positions:
            if start_pos in visited:
                continue

            # BFS flood-fill from this cell
            cluster_cells: Set[Tuple[int, int]] = set()
            total_value = 0
            queue = deque([start_pos])

            min_r, min_c = start_pos
            max_r, max_c = start_pos

            while queue:
                r, c = queue.popleft()
                if (r, c) in visited:
                    continue
                if (r, c) not in cell_positions:
                    continue

                visited.add((r, c))
                cluster_cells.add((r, c))

                val = board.get_cell(r, c)
                if val is not None and val != 0:
                    total_value += val

                # Update bounds
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

                # Check 4-connected neighbors (adjacent cells)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cell_positions and (nr, nc) not in visited:
                        queue.append((nr, nc))

            if cluster_cells:
                clusters.append(CellCluster(
                    cells=cluster_cells,
                    bounds=(min_r, min_c, max_r, max_c),
                    total_value=total_value
                ))

        return clusters

    def get_search_regions(self) -> List[Tuple[int, int, int, int]]:
        """
        Get minimal bounding regions that contain all clusters.

        Returns list of (min_row, min_col, max_row, max_col) regions
        that should be searched for valid moves.
        """
        if not self.clusters:
            return []

        # For very sparse grids, use individual cluster bounds
        if self.density < 0.3:
            # Expand cluster bounds slightly to catch cross-cluster moves
            regions = []
            for cluster in self.clusters:
                min_r, min_c, max_r, max_c = cluster.bounds
                # Expand by 1 in each direction (capped by grid)
                regions.append((
                    max(0, min_r - 1),
                    max(0, min_c - 1),
                    max_r + 1,
                    max_c + 1
                ))
            return regions

        # For moderate density, use full grid
        return []  # Empty means use full grid search


@register_strategy
class SparseBeamSearchStrategy(SolverStrategy):
    """
    Beam search optimized for sparse grids with scattered cells.

    Enhancements over standard beam search:
    1. Adaptive Parameters - Increases beam_width and depth for sparse grids
    2. Sparse-Aware Heuristic - Rewards connectivity and respects move scarcity
    3. Cluster Prioritization - Prefers moves clearing entire isolated groups
    4. Spatial Index - Reduces search space by focusing on cell regions

    Use this strategy when grid density < 50% for better results.
    Falls back to standard beam search behavior for dense grids.
    """
    name = "sparse_beam"
    description = "Sparse Beam Search - Optimized for scattered cells"
    timeout_sec = 15.0

    # Density thresholds for parameter adaptation
    DENSE_THRESHOLD = 0.6      # Above this, use standard params
    MODERATE_THRESHOLD = 0.3   # Between this and dense, use moderate params
    # Below moderate = sparse, use aggressive params

    def __init__(
        self,
        base_depth: int = 2,
        base_beam_width: int = 5,
        max_depth: int = 4,
        max_beam_width: int = 12,
        connectivity_weight: float = 0.5,
        scarcity_weight: float = 2.0,
        cluster_bonus: float = 1.5
    ):
        """
        Initialize sparse beam search strategy.

        Args:
            base_depth: Minimum search depth (used for dense grids)
            base_beam_width: Minimum beam width (used for dense grids)
            max_depth: Maximum search depth (used for very sparse grids)
            max_beam_width: Maximum beam width (used for very sparse grids)
            connectivity_weight: Weight for connectivity improvement bonus
            scarcity_weight: Weight for move scarcity adjustment
            cluster_bonus: Bonus multiplier for clearing entire clusters
        """
        self.base_depth = base_depth
        self.base_beam_width = base_beam_width
        self.max_depth = max_depth
        self.max_beam_width = max_beam_width
        self.connectivity_weight = connectivity_weight
        self.scarcity_weight = scarcity_weight
        self.cluster_bonus = cluster_bonus

        # Runtime state
        self._current_depth = base_depth
        self._current_beam_width = base_beam_width
        self._spatial_index: Optional[SpatialIndex] = None

    def solve(self, context: SolutionContext) -> Solution:
        """
        Compute full solution using sparse-optimized beam search.

        Args:
            context: Solution context with board and cancellation

        Returns:
            Solution with moves and metrics
        """
        board = context.board
        total_cells = board.rows * board.cols
        initial_cells = board.count_cells()
        density = initial_cells / total_cells if total_cells > 0 else 0.0

        # Fast path: for dense grids, delegate to standard beam search
        # Sparse optimizations add overhead without benefit for dense grids
        if density >= self.DENSE_THRESHOLD:
            from .beam_search import BeamSearchStrategy
            logger.info(
                f"[SparseBeam] Dense grid ({density:.1%}), delegating to standard beam"
            )
            standard = BeamSearchStrategy(
                depth=self.base_depth,
                beam_width=self.base_beam_width
            )
            return standard.solve(context)

        start_time = time.perf_counter()

        moves: List[Move] = []
        board_states: List[BoardState] = [board]
        states_explored = 0
        total_cleared = 0

        # Build initial spatial index and adapt parameters
        self._spatial_index = SpatialIndex.build(board)
        self._adapt_parameters(self._spatial_index.density)

        logger.info(
            f"[SparseBeam] Grid density: {self._spatial_index.density:.1%}, "
            f"cells: {len(self._spatial_index.cell_positions)}, "
            f"using depth={self._current_depth}, beam_width={self._current_beam_width}"
        )

        while True:
            if self._check_cancelled(context):
                return self._build_solution(
                    moves, board_states, total_cleared, states_explored,
                    start_time, was_cancelled=True
                )

            # Rebuild spatial index for current board
            self._spatial_index = SpatialIndex.build(board)

            # Re-adapt parameters as board becomes sparser
            self._adapt_parameters(self._spatial_index.density)

            best_move, explored = self._select_best_move(board, context)
            states_explored += explored

            if best_move is None:
                break

            moves.append(best_move)
            total_cleared += best_move.cell_count
            board = board.apply_move(best_move)
            board_states.append(board)

            if initial_cells > 0:
                progress = min(0.99, total_cleared / initial_cells)
                context.report_progress(
                    progress,
                    f"{len(moves)} moves, {total_cleared} cells cleared"
                )

            logger.debug(
                f"[SparseBeam] Move {len(moves)}: clearing {best_move.cell_count} cells, "
                f"total {total_cleared}/{initial_cells}"
            )

        logger.info(
            f"[SparseBeam] Solution complete: {len(moves)} moves, "
            f"{total_cleared} cells, {states_explored} states explored"
        )

        return self._build_solution(
            moves, board_states, total_cleared, states_explored,
            start_time, was_cancelled=False
        )

    def _adapt_parameters(self, density: float) -> None:
        """
        Adapt beam width and depth based on grid density.

        Sparse grids need wider beams and deeper search to find
        the few valid moves and explore their consequences.
        """
        if density >= self.DENSE_THRESHOLD:
            # Dense grid: use base parameters
            self._current_depth = self.base_depth
            self._current_beam_width = self.base_beam_width
        elif density >= self.MODERATE_THRESHOLD:
            # Moderate density: interpolate
            factor = (self.DENSE_THRESHOLD - density) / (self.DENSE_THRESHOLD - self.MODERATE_THRESHOLD)
            self._current_depth = int(self.base_depth + factor * (self.max_depth - self.base_depth))
            self._current_beam_width = int(self.base_beam_width + factor * (self.max_beam_width - self.base_beam_width))
        else:
            # Sparse grid: use maximum parameters
            self._current_depth = self.max_depth
            self._current_beam_width = self.max_beam_width

    def _select_best_move(
        self,
        board: BoardState,
        context: SolutionContext
    ) -> Tuple[Optional[Move], int]:
        """
        Use sparse-optimized beam search to find best first move.
        """
        # Fast path: if very few moves exist, skip beam search overhead
        valid_moves = self._find_valid_moves_sparse(board)
        states_explored = len(valid_moves)

        if not valid_moves:
            return None, states_explored

        if len(valid_moves) <= 3:
            # Too few moves to benefit from beam search - just pick best
            best = max(valid_moves, key=lambda m: m.cell_count)
            return best, states_explored

        # Full beam search for boards with more move options
        root = SparseBeamNode(board=board, path=(), cleared=0)
        root.score = self._calculate_score(root, self._spatial_index)
        beam = [root]
        states_explored += 1

        for d in range(self._current_depth):
            if self._check_cancelled(context):
                break

            beam, explored = self._expand_beam(beam, context)
            states_explored += explored

            if not beam:
                break

        if beam and beam[0].path:
            return beam[0].path[0], states_explored

        # Fallback: simple greedy (shouldn't reach here normally)
        if valid_moves:
            best = max(valid_moves, key=lambda m: m.cell_count)
            return best, states_explored

        return None, states_explored

    def _expand_beam(
        self,
        beam: List[SparseBeamNode],
        context: SolutionContext
    ) -> Tuple[List[SparseBeamNode], int]:
        """
        Expand beam with cluster-aware move prioritization.
        """
        candidates: List[SparseBeamNode] = []
        states_explored = 0

        for node in beam:
            if self._check_cancelled(context):
                break

            # Rebuild spatial index for this node's board
            spatial_index = SpatialIndex.build(node.board)
            valid_moves = self._find_valid_moves_sparse(node.board, spatial_index)
            states_explored += len(valid_moves)

            # Score and sort moves with cluster awareness
            scored_moves = self._score_moves(valid_moves, node.board, spatial_index)
            scored_moves.sort(key=lambda x: x[1], reverse=True)

            for move, move_score in scored_moves:
                new_board = node.board.apply_move(move)

                # Check if this move cleared any complete clusters
                clusters_cleared = self._count_cleared_clusters(
                    move, spatial_index.clusters
                )

                new_node = SparseBeamNode(
                    board=new_board,
                    path=node.path + (move,),
                    cleared=node.cleared + move.cell_count,
                    clusters_cleared=node.clusters_cleared + clusters_cleared
                )
                # Use None to trigger fast density-based estimation
                # (new_board differs from spatial_index's board)
                new_node.score = self._calculate_score(new_node, None)
                candidates.append(new_node)

                if new_board.count_cells() == 0:
                    return [new_node], states_explored

        candidates.sort()
        return candidates[:self._current_beam_width], states_explored

    def _find_valid_moves_sparse(
        self,
        board: BoardState,
        spatial_index: Optional[SpatialIndex] = None
    ) -> List[Move]:
        """
        Find valid moves using spatial index to reduce search space.

        For sparse grids, only searches rectangles that intersect
        with known cell clusters, dramatically reducing O(n^4) overhead.
        """
        if spatial_index is None:
            spatial_index = self._spatial_index or SpatialIndex.build(board)

        # For very sparse grids, use optimized search
        if spatial_index.density < 0.4:
            return self._find_moves_cluster_focused(board, spatial_index)

        # For denser grids, use standard search
        return self.find_all_valid_moves(board)

    def _find_moves_cluster_focused(
        self,
        board: BoardState,
        spatial_index: SpatialIndex
    ) -> List[Move]:
        """
        Find moves by focusing on cluster regions only.

        Instead of O(rows^2 * cols^2), we search O(k * region_size^2)
        where k is the number of clusters.
        """
        moves: List[Move] = []
        seen_moves: Set[Tuple[int, int, int, int]] = set()

        # Get rows and columns that actually have cells
        active_rows = sorted(spatial_index.row_has_cells)
        active_cols = sorted(spatial_index.col_has_cells)

        if not active_rows or not active_cols:
            return moves

        # Only search rectangles that span active rows/columns
        for i1, r1 in enumerate(active_rows):
            for j1, c1 in enumerate(active_cols):
                for r2 in active_rows[i1:]:
                    for c2 in active_cols[j1:]:
                        # Skip if we've seen this rectangle
                        rect_key = (r1, c1, r2, c2)
                        if rect_key in seen_moves:
                            continue
                        seen_moves.add(rect_key)

                        # Calculate sum and collect cells
                        total = 0
                        cells = []

                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                val = board.get_cell(r, c)
                                if val is not None and val != 0:
                                    total += val
                                    cells.append((r, c))

                        if total == 10 and cells:
                            move = Move.create(r1, c1, r2, c2, cells, total)
                            moves.append(move)

        return moves

    def _score_moves(
        self,
        moves: List[Move],
        board: BoardState,
        spatial_index: SpatialIndex
    ) -> List[Tuple[Move, float]]:
        """
        Score moves with cluster and connectivity awareness.

        Prioritizes:
        1. Moves that clear more cells
        2. Moves that clear entire clusters (bonus)
        3. Moves that improve connectivity of remaining cells
        """
        scored: List[Tuple[Move, float]] = []

        for move in moves:
            score = float(move.cell_count)

            # Cluster clearing bonus
            clusters_cleared = self._count_cleared_clusters(move, spatial_index.clusters)
            score += clusters_cleared * self.cluster_bonus

            # Connectivity improvement (clearing isolated cells is good)
            isolation_score = self._calculate_isolation_score(move, spatial_index)
            score += isolation_score * self.connectivity_weight

            scored.append((move, score))

        return scored

    def _count_cleared_clusters(
        self,
        move: Move,
        clusters: List[CellCluster]
    ) -> int:
        """Count how many clusters are completely cleared by this move."""
        move_cells = set(move.cells)
        cleared_count = 0

        for cluster in clusters:
            if cluster.cells.issubset(move_cells):
                cleared_count += 1

        return cleared_count

    def _calculate_isolation_score(
        self,
        move: Move,
        spatial_index: SpatialIndex
    ) -> float:
        """
        Calculate bonus for clearing isolated cells.

        Cells with fewer neighbors are more isolated and harder to clear,
        so clearing them is more valuable.
        """
        score = 0.0

        for r, c in move.cells:
            # Count how many neighbors this cell has
            neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in spatial_index.cell_positions:
                    neighbors += 1

            # More isolated cells (fewer neighbors) get higher bonus
            isolation = 4 - neighbors  # 0-4 scale
            score += isolation * 0.25

        return score

    def _calculate_score(
        self,
        node: SparseBeamNode,
        spatial_index: Optional[SpatialIndex] = None
    ) -> float:
        """
        Calculate node score with sparse-aware heuristic.

        Score = path_score + connectivity_bonus + scarcity_estimate + cluster_bonus

        Uses density-based estimation instead of expensive move counting.

        Args:
            node: The beam node to score
            spatial_index: Pre-built spatial index (avoids redundant builds)
        """
        path_score = float(node.cleared)

        # Cluster clearing bonus
        cluster_bonus = node.clusters_cleared * self.cluster_bonus

        remaining_cells = node.board.count_cells()

        if remaining_cells == 0:
            # Board cleared - maximum score
            return path_score + cluster_bonus + 100.0

        # Use provided spatial_index or estimate from remaining cells
        if spatial_index is not None:
            cluster_count = len(spatial_index.clusters)
            density = spatial_index.density
        else:
            # Estimate without building full index
            total_cells = node.board.rows * node.board.cols
            density = remaining_cells / total_cells if total_cells > 0 else 0.0
            # Rough cluster estimate: fewer cells = more fragmented
            cluster_count = max(1, int(remaining_cells * (1 - density) * 0.5))

        # Scarcity estimate based on density (sparse = fewer moves likely)
        # Avoids expensive O(n^4) move enumeration
        scarcity_adjustment = 0.0
        if density < 0.5:
            # Sparser grids get higher scarcity bonus
            scarcity_adjustment = self.scarcity_weight * (0.5 - density)

        # Connectivity bonus: reward boards where remaining cells are clustered
        connectivity_bonus = 0.0
        if cluster_count > 0:
            avg_cluster_size = remaining_cells / cluster_count
            connectivity_bonus = avg_cluster_size * self.connectivity_weight * 0.1

        return path_score + cluster_bonus + scarcity_adjustment + connectivity_bonus

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
