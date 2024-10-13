from dataset_worker.settings.children_settings.dataset_preparer_settings import UnderDatasetPreparerSettings
from dataset_worker.settings.base_settings import SettingsInterface
from pandas import DataFrame


class SelectorSettings(SettingsInterface):
    def __init__(self, timestamp_id: int, under_dataset_settings: UnderDatasetPreparerSettings,
                 ticker_column_id: int = None):
        self.ticker_column_id = ticker_column_id
        self.under_dataset_settings = under_dataset_settings
        self.timestamp_id = timestamp_id

    def check_data_correctness(self, data: DataFrame):
        if self.ticker_column_id and (self.ticker_column_id >= len(data.columns) or self.ticker_column_id < 0):
            raise ValueError(f"ticker_column_id = {self.ticker_column_id} is out of bounds")

        if self.timestamp_id >= len(data.columns) or self.timestamp_id < 0:
            raise ValueError(f"timestamp_id = {self.timestamp_id} is out of bounds")