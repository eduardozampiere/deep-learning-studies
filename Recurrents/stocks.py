import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from keras.optimizers import SGD

from sklearn.preprocessing import MinMaxScaler


def plot_chart(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Fechamento')
    plt.ylabel('Preço', fontsize=10)
    plt.xlabel('Data', fontsize=10)
    plt.show()


class NeuralNetwork:
    def __init__(self, dataset: pd.DataFrame, model):
        self.dataset = dataset

        assert self.dataset.empty is False, "Dataset must not be empty"

        self.training_data = None
        self.test_data = None
        self.validation_data = None

        self.model = model

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.prices = self.dataset['Close']

        self.days_time_step = 15

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.x_validation = []
        self.y_validation = []

        self.history = None

        self.transform_data()

    def transform_data(self):
        self.remove_nulls()
        self.get_train_test()
        self.normalize_data()
        self.get_x_y()

    def get_x_y(self):
        for i in range(self.days_time_step, len(self.training_data)):
            self.x_train.append(self.training_data[i-self.days_time_step:i])
            self.y_train.append(self.training_data[i])

        for i in range(self.days_time_step, len(self.test_data)):
            # self.x_test.append(self.training_data[i-self.days_time_step:i])
            self.x_test.append(self.test_data[i-self.days_time_step:i])

        for i in range(self.days_time_step, len(self.validation_data)):
            self.x_validation.append(self.validation_data[i-self.days_time_step:i])
            self.y_validation.append(self.validation_data[i])

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_test, self.y_test = np.array(self.x_test), np.array(self.y_test)
        self.x_validation, self.y_validation = np.array(self.x_validation), np.array(self.y_validation)


    def get_train_test(self):
        training_size = int(self.prices.shape[0] * 0.95)
        self.training_data = np.array(self.prices[:training_size])
        self.validation_data = np.array(self.prices[training_size-self.days_time_step:])
        self.test_data = np.array(self.prices[training_size:])

    def normalize_data(self):
        self.training_data = self.scaler.fit_transform(self.training_data.reshape(-1, 1))
        self.test_data = self.scaler.transform(self.test_data.reshape(-1, 1))
        self.validation_data = self.scaler.transform(self.validation_data.reshape(-1, 1))

    def remove_nulls(self):
        self.dataset = self.dataset.dropna()
        null_rows = self.dataset[self.dataset.isnull().any(axis=1)]
        assert null_rows.shape[0] == 0, "There are still null rows"


    def fit_rnn(self):
        self.history = model.fit(self.x_train, self.y_train, epochs=30, batch_size=32, validation_data=(self.x_validation, self.y_validation))


    def show_loss_chart(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()


    def predict(self):
        predicted_stock_price = self.model.predict(self.x_test)
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)

        self.test_data = self.scaler.inverse_transform(self.test_data)

        plt.figure(figsize=(18, 9))
        plt.plot(self.test_data[self.days_time_step:], color='green', label='real')
        plt.plot(predicted_stock_price, color='red', label='previsão')
        plt.xlabel('Datas', fontsize=18)
        plt.ylabel('Preço', fontsize=18)
        plt.title("Projeção de Preço PETR4", fontsize=30)
        plt.legend()
        plt.show()

        return predicted_stock_price

def get_data(ticker, period='10y'):
    equity = yf.Ticker(ticker)
    return equity.history(period=period)


def create_rnn_mode(days_time_step=15):
    model = Sequential()
    model.add(SimpleRNN(units=100, input_shape=(days_time_step, 1), return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

if __name__ == '__main__':
    data = get_data('PETR4.SA', period='max')
    model = create_rnn_mode()
    nn = NeuralNetwork(data, model)
    nn.fit_rnn()

    k = nn.predict()
    l=0

    # plot_chart(nn.dataset)
