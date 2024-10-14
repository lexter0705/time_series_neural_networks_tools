from __future__ import annotations

import numpy
from dataset_worker.data_types.data_types import Dataset
from dataset_worker.splitters.base_splitter_children import YXSplitter, TestTrainSplitter
from dataset_worker.splitters.base_splitter import BaseSplitter
from dataset_worker.settings.children_settings.splitter_settings import DatasetSplitterSettings


class DatasetSplitter(BaseSplitter):
    def __init__(self, main_dataset: numpy.ndarray, settings: DatasetSplitterSettings):
        super().__init__(main_dataset)
        settings.check_data_correctness(main_dataset)
        self.__settings = settings
        self.__result: Dataset | None = None

    def __split_dataset_on_test_train(self, dataset) -> (numpy.ndarray, numpy.ndarray):
        test_train_splitter = TestTrainSplitter(dataset, self.__settings.test_dataset_percent)
        test_train_splitter.split_dataset()
        dataset = test_train_splitter.get_prepared_dataset()
        return dataset.x_train, dataset.x_test

    def __split_test_train_datasets(self):
        x_train, x_test = self.__split_dataset_on_test_train(self.__result.x_train)
        y_train, y_test = self.__split_dataset_on_test_train(self.__result.y_train)
        self.__result = Dataset(x_train, y_train, x_test, y_test)

    def __split_x_y_dataset(self):
        first_splitter = YXSplitter(self._get_main_dataset(), self.__settings.x_columns_id,
                                    self.__settings.y_columns_id)
        first_splitter.split_dataset()
        self.__result = first_splitter.get_prepared_dataset()

    def split_dataset(self):
        if self.__settings.x_columns_id and self.__settings.y_columns_id:
            self.__split_x_y_dataset()
        if self.__settings.test_dataset_percent:
            self.__split_test_train_datasets()

    def get_prepared_dataset(self) -> Dataset:
        return self.__result
