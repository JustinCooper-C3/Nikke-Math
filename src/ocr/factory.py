"""
OCR Engine Factory

Factory for creating OCR engine instances.
"""

from pathlib import Path
from typing import Dict, Type

from .base import OCREngine


# Registry of available engines
_ENGINE_REGISTRY: Dict[str, str] = {
    "template": "template_engine.TemplateOCREngine",
}

# Cache for loaded engine classes
_ENGINE_CACHE: Dict[str, Type[OCREngine]] = {}


def _load_engine_class(engine_type: str) -> Type[OCREngine]:
    """Lazily load an engine class by type."""
    if engine_type in _ENGINE_CACHE:
        return _ENGINE_CACHE[engine_type]

    module_class = _ENGINE_REGISTRY[engine_type]
    module_name, class_name = module_class.rsplit(".", 1)

    # Import the module dynamically
    import importlib
    module = importlib.import_module(f".{module_name}", package="src.ocr")
    engine_class = getattr(module, class_name)

    _ENGINE_CACHE[engine_type] = engine_class
    return engine_class


def create_engine(engine_type: str = "template", **config) -> OCREngine:
    """
    Create an OCR engine by type.

    Args:
        engine_type: Engine type identifier. Available types:
            - "template" (default): OpenCV template matching
        **config: Engine-specific configuration options:
            For "template":
                - template_dir: Path to custom digit templates

    Returns:
        Configured OCREngine instance

    Raises:
        ValueError: If engine_type is not recognized

    Example:
        # Create template engine
        engine = create_engine()

        # Create with custom templates
        engine = create_engine("template", template_dir="./my_templates")

        # Process an image
        result = engine.process(image)
        board = result.board  # 2D array of digits
    """
    if engine_type not in _ENGINE_REGISTRY:
        available = ", ".join(_ENGINE_REGISTRY.keys())
        raise ValueError(f"Unknown engine type: {engine_type}. Available: {available}")

    engine_class = _load_engine_class(engine_type)

    # Extract constructor args
    template_dir = config.pop("template_dir", None)
    if template_dir is not None:
        template_dir = Path(template_dir)
    engine = engine_class(template_dir=template_dir)

    # Apply remaining config
    if config:
        engine.configure(**config)

    return engine


def register_engine(name: str, engine_class: type) -> None:
    """
    Register a custom OCR engine type.

    Args:
        name: Engine type identifier
        engine_class: OCREngine subclass

    Example:
        from src.ocr import register_engine, OCREngine

        class MyCustomEngine(OCREngine):
            ...

        register_engine("custom", MyCustomEngine)
    """
    if not issubclass(engine_class, OCREngine):
        raise TypeError(f"{engine_class} must be a subclass of OCREngine")
    _ENGINE_REGISTRY[name] = engine_class


def available_engines() -> list[str]:
    """
    List available engine types.

    Returns:
        List of registered engine type names
    """
    return list(_ENGINE_REGISTRY.keys())
