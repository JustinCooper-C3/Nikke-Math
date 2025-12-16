"""
OCR Engine Base Interface

Abstract base class defining the OCR engine contract.
"""

from abc import ABC, abstractmethod
from PIL import Image

from .result import OCRResult


class OCREngine(ABC):
    """
    Abstract base class for OCR engines.

    All OCR implementations must inherit from this class and implement
    the process() method to extract game board values from an image.
    """

    @abstractmethod
    def process(self, image: Image.Image) -> OCRResult:
        """
        Process an image and extract the game board.

        Args:
            image: PIL Image of the game window

        Returns:
            OCRResult containing:
            - board: List[List[Optional[int]]] - 2D array of digits 1-9 or None
            - confidence: float - average confidence score
            - grid_info: GridInfo - cell positions for overlay rendering
            - cell_results: List[CellResult] - per-cell details
            - uncertain_count: int - cells below confidence threshold
            - total_cells: int - total cells detected
            - processing_time_ms: float - processing duration
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Engine identifier.

        Returns:
            String name identifying this engine type (e.g., "template", "tesseract")
        """
        pass

    def configure(self, **kwargs) -> None:
        """
        Configure engine parameters.

        Override in subclasses to support runtime configuration.
        Default implementation does nothing.

        Args:
            **kwargs: Engine-specific configuration options
        """
        pass
