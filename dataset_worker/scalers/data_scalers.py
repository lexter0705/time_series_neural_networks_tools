from __future__ import annotations
from typing import Type

import numpy
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


class ManyColumnsScaler:
    def __init__(self):
        self.__scalers: list[MinMaxScaler] = []

    def fit(self, data: ndarray):
        for i in range(len(data[0])):
            scaler = MinMaxScaler()
            scaler.fit(numpy.reshape(data[:, i], (-1, 1)))
            self.__scalers.append(scaler)

    def transform(self, data: ndarray) -> ndarray:
        returned_data = numpy.empty(data.shape)
        for i in range(len(self.__scalers)):
            transformed_data = self.__scalers[i].transform(numpy.reshape(data[:, i], (-1, 1)))
            transformed_data = numpy.reshape(transformed_data, transformed_data.shape[0])
            returned_data[:, i] = transformed_data

        return returned_data

    def inverse_transform(self, data: ndarray) -> ndarray:
        returned_data = numpy.empty(data.shape)
        for i in range(len(self.__scalers)):
            transformed_data = self.__scalers[i].inverse_transform(numpy.reshape(data[:, i], (-1, 1)))
            transformed_data = numpy.reshape(transformed_data, transformed_data.shape[0])
            returned_data[:, i] = transformed_data

        return returned_data


class DataScalers:
    def __init__(self, x_scaler: MinMaxScaler | ManyColumnsScaler = None,
                 y_scaler: MinMaxScaler | ManyColumnsScaler = None,
                 scaler_type: Type[MinMaxScaler] | Type[ManyColumnsScaler] = None):
        if not ((x_scaler and y_scaler) or scaler_type):
            raise Exception("x_scaler and y_scaler or scaler_type should be filled")

        self.__x_scaler = x_scaler if x_scaler else scaler_type()
        self.__y_scaler = y_scaler if x_scaler else scaler_type()

    def fit_scalers(self, x_dataset: DataFrame | ndarray, y_dataset: DataFrame | ndarray):
        self.__x_scaler.fit(x_dataset)
        self.__y_scaler.fit(y_dataset)

    def get_x_scaler(self) -> MinMaxScaler | ManyColumnsScaler:
        return self.__x_scaler

    def get_y_scaler(self) -> MinMaxScaler | ManyColumnsScaler:
        return self.__y_scaler
