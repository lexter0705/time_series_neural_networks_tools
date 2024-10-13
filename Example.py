from pandas.core.interchange.dataframe_protocol import DataFrame

from neural.creator.neural_creator import LSTMLayerNeural
from dataset_worker.dataset_selector import DatasetSelector
from dataset_worker.preparers.dataset_preparer import DatasetPreparer
from dataset_worker.scalers.scalers_fitter import ScalersFitter
from dataset_worker.settings.children_settings.splitter_settings import DatasetSplitterSettings
from dataset_worker.settings.children_settings.selector_settings import SelectorSettings
from dataset_worker.settings.children_settings.dataset_preparer_settings import UnderDatasetPreparerSettings, \
    PreparerSettings
from dataset_worker.scalers.data_scalers import ManyColumnsScaler
import pandas as pd


def dataset_to_timestamp(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["DATE-TIME"] = pd.to_datetime(dataset["DATE"] + " " + dataset["TIME"], infer_datetime_format=True)
    dataset["timestamp"] = dataset["DATE-TIME"].apply(lambda x: x.timestamp())
    dataset = dataset.drop(["DATE", "TIME", "DATE-TIME"], axis=1)
    return dataset


def fit_model(selector_settings: SelectorSettings, data: DataFrame, name: str):
    selector = DatasetSelector(data, selector_settings)
    selector.select_under_datasets()
    datasets = selector.get_all_datasets()
    neural_creator = LSTMLayerNeural(datasets)
    neural_creator.create_model()
    neural_creator.fit(name)


def create_neural_network():
    ai_name = "neural_networks/last_model_9_4"
    csv_file_path = "../data/train.csv"
    data = pd.read_csv(csv_file_path, sep=";")
    data = dataset_to_timestamp(data)

    preparer_settings = PreparerSettings(columns_for_delete=["PER", "TICKER"])
    preparer = DatasetPreparer(data, preparer_settings)
    preparer.preparing_dataset()
    data = preparer.get_dataset()
    scaler_fitter = ScalersFitter(data.to_numpy(), [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4], ManyColumnsScaler)
    scaler_fitter.fit_scalers()

    splitter_settings = DatasetSplitterSettings(x_columns_id=[0, 1, 2, 3, 4, 5], y_columns_id=[0, 1, 2, 3, 4])
    under_dataset_settings = UnderDatasetPreparerSettings(scaler_fitter.get_scalers(), 70, splitter_settings)
    selector_settings = SelectorSettings(5, under_dataset_settings)

    fit_model(selector_settings, data, ai_name)


if __name__ == "__main__":
    create_neural_network()
