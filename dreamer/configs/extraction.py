from dataclasses import dataclass
from .configurable import Configurable


@dataclass
class ExtractionConfig(Configurable):
    """
    Extraction stage configurations
    """
    PATH_TO_SEARCHABLES: str = 'searchables'
    PARALLELIZE: bool = True
    BASE_EDGE_LENGTH: int = 2


extraction_config: ExtractionConfig = ExtractionConfig()
