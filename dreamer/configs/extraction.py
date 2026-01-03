from dataclasses import dataclass
from .configurable import Configurable


@dataclass
class ExtractionConfig(Configurable):
    """
    Extraction stage configurations
    """
    PATH_TO_SEARCHABLES: str = 'searchables'


extraction_config: ExtractionConfig = ExtractionConfig()
