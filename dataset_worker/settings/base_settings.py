from __future__ import annotations

from numpy import ndarray
from pandas import DataFrame
import abc


class SettingsInterface(abc.ABC):
    @abc.abstractmethod
    def check_data_correctness(self, data: DataFrame | ndarray):
        pass
