from numpy import ndarray
from dataset_worker.settings.base_settings import SettingsInterface


class DatasetSplitterSettings(SettingsInterface):
    def __init__(self, test_dataset_percent: float = None, x_columns_id: list[int] = None,
                 y_columns_id: list[int] = None, x_dataset: ndarray = None, y_dataset: ndarray = None):
        self.test_dataset_percent = test_dataset_percent
        self.x_columns_id = x_columns_id
        self.y_columns_id = y_columns_id
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset

    def check_data_correctness(self, data: ndarray):
        if self.test_dataset_percent and (self.test_dataset_percent < 0 or self.test_dataset_percent > 1):
            raise ValueError('test_dataset_percent must be between 0 and 1')

        if self.x_columns_id and not all([i in range(data.shape[0]) for i in self.x_columns_id]):
            raise ValueError("In x_columns_id, all columns must be present in the dataset")

        if self.y_columns_id and not all([i in range(data.shape[0]) for i in self.y_columns_id]):
            raise ValueError("In y_columns_id, all columns must be present in the dataset")