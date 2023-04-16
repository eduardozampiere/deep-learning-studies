from trainning_ml import NeuralNetwork
import pandas as pd


if __name__ == '__main__':
    # trainning_data, trainning_labels = create_trainning_data()
    data = pd.read_csv('src/diabetes.csv').values
    trainning_data = data[:, :-1]
    trainning_labels = data[:, -1]

    model = NeuralNetwork(trainning_data, trainning_labels, _lambda=0.15)
    model.train(epochs=100_000)



