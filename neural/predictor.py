from __future__ import annotations

import numpy
from dataset_worker.scalers.data_scalers import DataScalers
import numpy as np
from tensorflow.keras.models import Model, load_model


class NeuralPredictor:
    def __init__(self, scaler: DataScalers):
        self.__model: Model | None = None
        self.__scaler = scaler
        self.__result: numpy.ndarray | None = None

    def __save_predict(self, predict: numpy.ndarray, time, ticker):
        predicted = self.__scaler.get_y_scaler().inverse_transform(predict)
        predicted = np.append(predicted, time)
        if ticker:
            predicted = np.insert(predicted, 0, ticker)
        predicted = self.__scaler.get_x_scaler().transform(np.reshape(predicted, (-1, *predicted.shape)))
        self.result = numpy.vstack([self.__result, predicted])

    def __get_start_and_end_time(self, data: numpy.ndarray, count_minutes: int) -> (int, int):
        data = self.__scaler.get_x_scaler().inverse_transform(data)
        start_date = data[len(data) - 1]
        start_date = start_date[len(start_date) - 1]
        end_date = start_date + count_minutes * 60
        return int(start_date), int(end_date)

    def predict(self, data: numpy.ndarray, count_minutes, ticker: int = None):
        last_data = data
        self.__result = numpy.empty((1, last_data.shape[1]))
        start_time, end_time = self.__get_start_and_end_time(last_data, count_minutes)
        last_data = np.reshape(last_data, (-1, last_data.shape[0], last_data.shape[1]))
        for i in range(start_time, end_time, 60):
            predicted = self.__model.predict(last_data)
            self.__save_predict(predicted, i + 60, ticker)
            last_data[0] = np.vstack([last_data[0][1:], self.result[len(self.result) - 1]])

    def load_model(self, name: str):
        self.__model = load_model(name)

    def get_result(self) -> numpy.ndarray:
        return self.__scaler.get_x_scaler().inverse_transform(self.result)
