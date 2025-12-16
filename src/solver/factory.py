"""
Strategy Factory Module - Registry and factory for strategy instantiation.
"""

from typing import Dict, List, Type, Any

from .base import SolverStrategy


# Global registry of strategies
_STRATEGIES: Dict[str, Type[SolverStrategy]] = {}


def register_strategy(cls: Type[SolverStrategy]) -> Type[SolverStrategy]:
    """
    Decorator to register a strategy class.

    Usage:
        @register_strategy
        class MyStrategy(SolverStrategy):
            name = "my_strategy"
            ...

    Args:
        cls: Strategy class to register

    Returns:
        The same class (for decorator chaining)
    """
    _STRATEGIES[cls.name] = cls
    return cls


def create_strategy(name: str, **kwargs: Any) -> SolverStrategy:
    """
    Create a strategy instance by name.

    Args:
        name: Strategy name (e.g., "greedy", "lookahead")
        **kwargs: Additional arguments passed to strategy constructor

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy name not found
    """
    if name not in _STRATEGIES:
        available = ", ".join(_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return _STRATEGIES[name](**kwargs)


def get_strategy_names() -> List[str]:
    """
    Get list of available strategy names.

    Returns:
        List of registered strategy names
    """
    return list(_STRATEGIES.keys())


def get_strategy_info() -> List[Dict[str, str]]:
    """
    Get name and description for all registered strategies.

    Returns:
        List of dicts with 'name' and 'description' keys
    """
    return [
        {"name": cls.name, "description": cls.description}
        for cls in _STRATEGIES.values()
    ]


def get_default_strategy_name() -> str:
    """
    Get the default strategy name.

    Returns:
        Default strategy name ("greedy" if available, else first registered)
    """
    if "greedy" in _STRATEGIES:
        return "greedy"
    if _STRATEGIES:
        return next(iter(_STRATEGIES.keys()))
    return ""
