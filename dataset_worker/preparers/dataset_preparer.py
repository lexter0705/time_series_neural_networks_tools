from pandas import DataFrame
from dataset_worker.settings.children_settings.dataset_preparer_settings import PreparerSettings


class DatasetPreparer:
    def __init__(self, dataset: DataFrame, settings: PreparerSettings):
        settings.check_data_correctness(dataset)
        self.__categorical_columns = settings.categorical_columns
        self.__columns_for_delete = settings.columns_for_delete
        self.__categorical_data = settings.categorical_data
        self.__data = dataset

    def __delete_empty_string(self):
        self.__data = self.__data.dropna()

    def __replace_categorical_column_to_number(self, column_id: int):
        column_name = self.__categorical_columns[column_id]
        categorical = self.__get_categorical_names(column_id)
        self.__data[column_name] = self.__data[column_name].replace(categorical)

    def __delete_columns(self):
        if not self.__columns_for_delete:
            return

        self.__data = self.__data.drop(self.__columns_for_delete, axis=1)

    def __replace_categorical_columns_to_number(self):
        if not self.__categorical_columns:
            return

        for i in range(len(self.__categorical_columns)):
            self.__replace_categorical_column_to_number(i)

    def __get_categorical_names(self, column_id) -> dict:
        if self.__categorical_data and self.__categorical_data[column_id]:
            return self.__categorical_data[column_id]

        all_categorical = self.__data[self.__categorical_columns[column_id]].value_counts().index.to_list()
        categorical_inverse_dict = {}
        for i in range(len(all_categorical)):
            categorical_inverse_dict[all_categorical[i]] = i
        return categorical_inverse_dict

    def preparing_dataset(self):
        self.__delete_columns()
        self.__replace_categorical_columns_to_number()
        self.__delete_empty_string()

    def get_dataset(self) -> DataFrame:
        return self.__data
