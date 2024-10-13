from __future__ import annotations
import abc
import numpy
from dataset_worker.data_types.data_types import Dataset


class BaseSplitter(abc.ABC):
    def __init__(self, dataset: numpy.ndarray):
        if len(dataset) == 0:
            raise Exception("dataset is null")

        self.__data = dataset

    @abc.abstractmethod
    def split_dataset(self):
        pass

    @abc.abstractmethod
    def get_prepared_dataset(self) -> Dataset:
        pass

    def _get_main_dataset(self) -> numpy.ndarray:
        return self.__data
