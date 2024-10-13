from __future__ import annotations

from typing import Type
from dataset_worker.splitters.dataset_splitter import YXSplitter
import numpy
from dataset_worker.scalers.data_scalers import DataScalers, ManyColumnsScaler, MinMaxScaler


class ScalersFitter(YXSplitter):
    def __init__(self, data: numpy.ndarray, x_columns_id: list, y_columns_id: list,
                 scaler_type: Type[ManyColumnsScaler] | Type[MinMaxScaler]):
        super().__init__(data, x_columns_id, y_columns_id)
        self.__scalers = DataScalers(scaler_type=scaler_type)

    def fit_scalers(self):
        self.split_dataset()
        result = self.get_prepared_dataset()
        self.__scalers.fit_scalers(result.x_train, result.y_train)

    def get_scalers(self) -> DataScalers:
        return self.__scalers
