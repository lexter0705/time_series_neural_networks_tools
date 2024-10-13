from __future__ import annotations

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM
from neural.creator.base_creator import NeuralCreator
from dataset_worker.data_types.data_types import Dataset


class LSTMLayerNeural(NeuralCreator):
    def __init__(self, datasets: list[Dataset]):
        super().__init__(datasets)

    def create_model(self):
        data_input = Input(shape=(self.get_datasets()[0].x_train.shape[1], self.get_datasets()[0].x_train.shape[2]))
        lstm_way = LSTM(145, return_sequences="True")(data_input)
        lstm_way = LSTM(145, return_sequences="True")(lstm_way)
        lstm_way = LSTM(145, return_sequences="True")(lstm_way)
        lstm_way = Flatten()(lstm_way)
        fin_way = Dense(10, activation="linear")(lstm_way)
        fin_way = Dense(self.get_datasets()[0].y_train.shape[1], activation="linear")(fin_way)
        model = Model(data_input, fin_way)
        model.compile(loss="mse", optimizer=Adam(lr=1e-4))
        self.set_model(model)
