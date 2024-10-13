from __future__ import annotations
from typing import Callable

import numpy
from dataset_worker.settings.children_settings.dataset_preparer_settings import UnderDatasetPreparerSettings
from dataset_worker.splitters.dataset_splitter import DatasetSplitter
from dataset_worker.ironer import Ironer
from dataset_worker.data_types.data_types import Dataset


class UnderDatasetPreparer:
    def __init__(self, main_dataset: numpy.ndarray, settings: UnderDatasetPreparerSettings):
        settings.check_data_correctness(main_dataset)
        self.__main_dataset = main_dataset
        self.__settings = settings
        self.__result: Dataset | None = None

    def __split_dataset(self):
        splitter = DatasetSplitter(self.__main_dataset, self.__settings.splitter_settings)
        splitter.split_dataset()
        self.__result = splitter.get_prepared_dataset()

    def __set_result_by_method(self, method: Callable):
        if self.__settings.splitter_settings.test_dataset_percent:
            self.__result.x_test, self.__result.y_test = method(self.__result.x_test,
                                                                self.__result.y_test)

        self.__result.x_train, self.__result.y_train = method(self.__result.x_train,
                                                              self.__result.y_train)

    def __regeneration_columns(self, x_column: numpy.ndarray, y_column: numpy.ndarray):
        last_frame = x_column[:self.__settings.length_one_frame]
        new_x_column = numpy.reshape(last_frame, (-1, *last_frame.shape))
        x_column = x_column[self.__settings.length_one_frame:]
        y_column = y_column[self.__settings.length_one_frame - 1:]
        for i in x_column:
            last_frame = last_frame[1:]
            last_frame = numpy.vstack([last_frame, i])
            new_x_column = numpy.vstack([new_x_column, numpy.reshape(last_frame, (-1, *last_frame.shape))])
        return new_x_column, y_column

    def __rescale_column(self, x_column: numpy.ndarray, y_column: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        x_column = self.__settings.scalers.get_x_scaler().transform(x_column)
        y_column = self.__settings.scalers.get_y_scaler().transform(y_column)
        return x_column, y_column

    def __y_smooth_out(self):
        if not self.__settings.window_size:
            return
        ironer = Ironer(self.__result.y_train, self.__settings.window_size, self.__settings.columns_id_not_for_iron)
        ironer.smooth_out_dataset()
        self.__result.y_train = ironer.get_dataset()

    def __shift_close_column(self, x_column: numpy.ndarray, y_column: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        shift_coefficient = self.__settings.shift_coefficient
        y_column = numpy.roll(y_column, -1 * shift_coefficient, axis=0)
        y_column = numpy.delete(y_column, len(y_column) - shift_coefficient, axis=0)
        x_column = numpy.delete(x_column, len(x_column) - shift_coefficient, axis=0)
        return x_column, y_column

    def prepare_dataset(self):
        self.__split_dataset()
        self.__y_smooth_out()
        self.__set_result_by_method(self.__shift_close_column)
        self.__set_result_by_method(self.__rescale_column)
        self.__set_result_by_method(self.__regeneration_columns)

    def get_dataset(self) -> Dataset:
        return self.__result
