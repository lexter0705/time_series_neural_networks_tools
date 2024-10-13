from __future__ import annotations

import abc

from tensorflow.keras.models import Model
from dataset_worker.data_types.data_types import Dataset


class NeuralCreator(abc.ABC):
    def __init__(self, datasets: list[Dataset]):
        self.__datasets = datasets
        self.__model: Model | None = None

    @abc.abstractmethod
    def create_model(self):
        pass

    def fit(self, model_name_for_save: str):
        for i in range(len(self.__datasets)):
            self.__model.fit(self.__datasets[i].x_train, self.__datasets[i].y_train, epochs=30)
        self.__model.save(model_name_for_save)

    def get_datasets(self) -> list[Dataset]:
        return self.__datasets

    def set_model(self, model: Model):
        self.__model = model
