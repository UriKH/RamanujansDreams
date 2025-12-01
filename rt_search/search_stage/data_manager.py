from rt_search.utils.types import *

from dataclasses import dataclass, field
from ramanujantools import Matrix
import pandas as pd
from collections import UserDict


@dataclass(frozen=True)
class SearchVector:
    """
    A class representing a search vector in a specific space
    """
    start: Position
    trajectory: Position

    def __hash__(self):
        return hash((self.start, self.trajectory))


@dataclass
class SearchData:
    """
    A class representing a search data alongside a specific search vector
    """
    sv: SearchVector
    limit: float = None
    delta: float | str = None
    eigen_values: Dict = field(default_factory=dict)
    gcd_slope: float | None = None
    initial_values: Matrix = None
    LIReC_identify: bool = False
    errors: Dict[str, Exception | None] = field(default_factory=dict)


class DataManager(UserDict[SearchVector, SearchData]):
    """
    DataManager represents a set of results found in a specific search in a CMF
    """

    def __init__(self, use_LIReC: bool):
        super().__init__()
        self.use_LIReC = use_LIReC

    @property
    def identified_percentage(self) -> float:
        """
        Computes the percentage identified by the search vector, if no data collected mark as 1 (i.e. 100%)
        :return: The percentage identified by the search vector as a number in [0, 1]
        """
        df = self.as_df()
        if df is None:
            return 1
        if self.use_LIReC:
            frac = df['LIReC_identify'].sum() / len(df['LIReC_identify'])
        else:
            frac = 1 - df['initial_values'].isna().sum() / len(df['initial_values'])
        return frac

    @property
    def best_delta(self) -> Tuple[Optional[float], Optional[SearchVector]]:
        """
        The best delta found
        :return: A tuple of the delta value and the search vector it was found in.
        """
        df = self.as_df()
        if df.empty:
            return None, None

        deltas = df['delta'].dropna()
        if deltas.empty:
            return None, None

        row = df.loc[deltas.idxmax()]
        return row['delta'], row['sv']

    def get_data(self) -> List[SearchData]:
        """
        Gather all search data in the manager into a list
        :return: The data collected as a list
        """
        return list(self.values())

    def as_df(self) -> pd.DataFrame:
        """
        Convert the data into a dataframe
        :return: The pandas dataframe.
        """
        rows = [
            {
                "sv": sv,
                "delta": data.delta,
                "limit": data.limit,
                "eigen_values": data.eigen_values,
                "gcd_slope": data.gcd_slope,
                "initial_values": data.initial_values,
                "LIReC_identify": data.LIReC_identify,
                "errors": data.errors,
            }
            for sv, data in self.items()
        ]
        return pd.DataFrame(rows)
