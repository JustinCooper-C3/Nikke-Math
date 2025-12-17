"""
Strategies Package - Concrete strategy implementations.

Import this module to register all built-in strategies.
"""

from .greedy import GreedyStrategy
from .greedy_plus import GreedyPlusStrategy
from .beam_search import BeamSearchStrategy

__all__ = [
    "GreedyStrategy",
    "GreedyPlusStrategy",
    "BeamSearchStrategy",
]
