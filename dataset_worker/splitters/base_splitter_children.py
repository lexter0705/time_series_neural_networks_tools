from __future__ import annotations

import numpy
from dataset_worker.data_types.data_types import Dataset
from dataset_worker.splitters.base_splitter import BaseSplitter


class YXSplitter(BaseSplitter):
    def __init__(self, main_dataset: numpy.ndarray, x_columns_id: list[int],
                 y_columns_id: list[int]):
        if not x_columns_id:
            raise Exception("x_columns_id cant be empty")

        if not y_columns_id:
            raise Exception("y_columns_id cant be empty")

        if not all([0 <= i < len(main_dataset) for i in x_columns_id]):
            raise Exception("unexpected column in categorical_columns")

        if not all([0 <= i < len(main_dataset) for i in y_columns_id]):
            raise Exception("unexpected column in categorical_columns")

        BaseSplitter.__init__(self, main_dataset)
        self.__x_columns_id = x_columns_id
        self.__y_columns_id = y_columns_id
        self.__X: numpy.ndarray | None = None
        self.__Y: numpy.ndarray | None = None
        self.__result: Dataset | None = None

    def __split_x_y_dataset(self):
        main_dataset = self._get_main_dataset()
        self.__X: numpy.ndarray = numpy.empty((main_dataset.shape[0], len(self.__x_columns_id)))
        self.__Y: numpy.ndarray = numpy.empty((main_dataset.shape[0], len(self.__y_columns_id)))
        for i in range(len(self.__y_columns_id)):
            self.__Y[:, i] = main_dataset[:, self.__y_columns_id[i]]
        for i in range(len(self.__x_columns_id)):
            self.__X[:, i] = main_dataset[:, self.__x_columns_id[i]]
        self.__result = Dataset(self.__X, self.__Y, numpy.empty(0), numpy.empty(0))

    def split_dataset(self):
        self.__split_x_y_dataset()

    def get_prepared_dataset(self) -> Dataset:
        return self.__result


class TestTrainSplitter(BaseSplitter):
    def __init__(self, dataset: numpy.ndarray, test_dataset_percent):
        if 1 > test_dataset_percent < 0:
            raise Exception(f"test_dataset_percent uncorrectable: {test_dataset_percent}")

        super().__init__(dataset)
        self.__test_dataset_percent = test_dataset_percent
        self.__result: Dataset | None = None

    def __split_test_train_datasets(self):
        train, test = self.__split_test_train_dataset(self._get_main_dataset())
        self.result = Dataset(train, numpy.empty(0), test, numpy.empty(0))

    def __split_test_train_dataset(self, dataset: numpy.ndarray):
        deleted_length = len(dataset) * self.__test_dataset_percent
        deleted_length = round(deleted_length)
        train = dataset[0:len(dataset) - deleted_length]
        test = dataset[len(dataset) - deleted_length:]
        return train, test

    def split_dataset(self):
        self.__split_test_train_datasets()

    def get_prepared_dataset(self) -> Dataset:
        return self.__result
