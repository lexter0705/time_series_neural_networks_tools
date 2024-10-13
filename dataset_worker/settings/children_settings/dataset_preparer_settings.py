from numpy import ndarray
from pandas import DataFrame
from dataset_worker.scalers.data_scalers import DataScalers
from dataset_worker.settings.base_settings import SettingsInterface
from dataset_worker.settings.children_settings.splitter_settings import DatasetSplitterSettings


class UnderDatasetPreparerSettings(SettingsInterface):
    def __init__(self, scalers: DataScalers, length_one_frame: int, splitter_settings: DatasetSplitterSettings,
                 window_size: int = None, columns_id_not_for_iron: list[int] = None, shift_coefficient: int = 1):
        self.scalers = scalers
        self.length_one_frame = length_one_frame
        self.splitter_settings = splitter_settings
        self.window_size = window_size
        self.columns_id_not_for_iron = columns_id_not_for_iron
        self.shift_coefficient = shift_coefficient

    def check_data_correctness(self, dataset: ndarray):
        if self.length_one_frame <= 0 or self.length_one_frame > dataset.shape[0]:
            raise ValueError("Length one frame must be greater than or equal to zero")

        if self.window_size and (self.window_size <= 0 or self.window_size >= len(dataset)):
            raise ValueError("Window size must be greater than or equal to zero")


class PreparerSettings(SettingsInterface):
    def __init__(self, categorical_columns: list[str] = None,
                 categorical_data: list[dict] = None,
                 columns_for_delete: list[str] = None):
        self.categorical_columns = categorical_columns
        self.columns_for_delete = columns_for_delete
        self.categorical_data = categorical_data

    def check_data_correctness(self, data: DataFrame):
        columns = data.columns.to_list()

        if self.categorical_columns and not all([i in columns for i in self.categorical_columns]):
            raise ValueError("In categorical columns, all columns must be present in the dataset")

        if self.columns_for_delete and not all([i in columns for i in self.columns_for_delete]):
            raise ValueError("In columns for delete, all columns must be present in the dataset")
