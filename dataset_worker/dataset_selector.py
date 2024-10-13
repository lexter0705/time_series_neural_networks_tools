from __future__ import annotations

import numpy
import numpy as np
from pandas import DataFrame
from dataset_worker.data_types.data_types import Dataset
from dataset_worker.settings.children_settings.selector_settings import SelectorSettings
from dataset_worker.preparers.under_dataset_preparer import UnderDatasetPreparer


class DatasetSelector:
    def __init__(self, dataset: DataFrame, settings: SelectorSettings):
        settings.check_data_correctness(dataset)
        self.__data = dataset
        self.__settings = settings
        self.__datasets: list = []
        self.__under_datasets: list[Dataset] = []

    def __select_dataset_by_ticker(self) -> list:
        ticker_name = self.__data.columns.values.tolist()[self.__settings.ticker_column_id]
        under_datasets_list = []
        all_categorical = self.__data[ticker_name].value_counts().index.to_list()
        for i in all_categorical:
            under_dataset = self.__data[self.__data[ticker_name] == i]
            under_datasets_list.append(under_dataset)
        return under_datasets_list

    def __select_from_many_datasets_by_time(self, under_datasets: list[DataFrame]):
        for dataset in under_datasets:
            dataset = np.array(dataset)
            self.__select_from_one_dataset(dataset)

    def __select_from_one_dataset(self, dataset):
        under_dataset = []
        timestamp_id = self.__settings.timestamp_id
        last_timestamp = dataset[0][timestamp_id]
        for data in dataset:
            if data[timestamp_id] - last_timestamp <= 60:
                under_dataset.append(data)
            else:
                under_dataset = numpy.array(under_dataset)
                self.__datasets.append(under_dataset)
                under_dataset = []
            last_timestamp = data[timestamp_id]

    def __convert_datasets_to_under_datasets(self):
        settings = self.__settings.under_dataset_settings
        for i in self.__datasets:
            if len(i) < settings.length_one_frame:
                continue
            under_dataset_preparer = UnderDatasetPreparer(i, settings)
            under_dataset_preparer.prepare_dataset()
            under_dataset = under_dataset_preparer.get_dataset()
            self.__under_datasets.append(under_dataset)

    def select_under_datasets(self):
        if self.__settings.ticker_column_id:
            selected_by_tickers = self.__select_dataset_by_ticker()
        else:
            selected_by_tickers = [self.__data.to_numpy().tolist()]

        self.__select_from_many_datasets_by_time(selected_by_tickers)
        self.__convert_datasets_to_under_datasets()

    def get_all_datasets(self) -> list[Dataset]:
        return self.__under_datasets
