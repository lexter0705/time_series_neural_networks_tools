import numpy


class Ironer:
    def __init__(self, data: numpy.ndarray, window_size: int, not_for_iron_columns_id: list[int] = None):
        if not data:
            raise Exception("data cant be empty")

        if window_size <= 0:
            raise Exception("window_size cant be <= 0>")

        self.__data = data
        self.__window_size = window_size
        self.__not_for_iron_columns_id = not_for_iron_columns_id if not_for_iron_columns_id else []

    def __smooth_out_column(self, column_id: int):
        column = numpy.empty(self.__data[:, column_id].shape)
        column_for_math = self.__data[:, column_id]
        for i in range(self.__window_size, len(column) + 1):
            median = numpy.median(column_for_math[(i - self.__window_size):i])
            column[i - 1] = median

        self.__data[:, column_id] = column

    def smooth_out_dataset(self):
        columns = range(self.__data.shape[1])
        columns = [i for i in columns if i not in self.__not_for_iron_columns_id]
        for i in columns:
            self.__smooth_out_column(i)

    def get_dataset(self) -> numpy.ndarray:
        return self.__data
