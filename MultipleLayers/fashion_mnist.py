import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


class NeuralNetwork:
    def __init__(self, dataset_train, dataset_test, model):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test

        assert self.dataset_train.empty is False, "Dataset must not be empty"
        assert self.dataset_test.empty is False, "Test dataset must not be empty"

        assert model, "Model must not be None"
        self.model = model

        self.standard_scaler = StandardScaler()

        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None

        self.init_treatment()

    def init_treatment(self):
        self.remove_nulls()
        self.get_x_y()
        self.normalize_data()

    def get_x_y(self):
        self.x = self.dataset_train.iloc[:, 1:]
        self.y = self.dataset_train.iloc[:, 0]
        self.x_test = self.dataset_test.iloc[:, 1:]
        self.y_test = self.dataset_test.iloc[:, 0]

        assert self.x.shape[0] == self.y.shape[0], "X and Y must have the same number of rows"
        assert self.x_test.shape[0] == self.y_test.shape[0], "X test and Y test must have the same number of rows"

    def remove_nulls(self):
        self.dataset_train = self.dataset_train.dropna()
        self.dataset_test = self.dataset_test.dropna()

        null_rows = self.dataset_train[self.dataset_train.isnull().any(axis=1)]
        assert null_rows.shape[0] == 0, "There are still null rows in train dataset"

        null_rows = self.dataset_test[self.dataset_test.isnull().any(axis=1)]
        assert null_rows.shape[0] == 0, "There are still null rows in test dataset"

    def normalize_data(self):
        self.x = self.standard_scaler.fit_transform(self.x)
        self.x_test = self.standard_scaler.fit_transform(self.x_test)


    def fit(self, batch_size=32, epochs=5):
        self.model.fit(self.x, self.y, batch_size=batch_size, epochs=epochs)

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test)

def create_model_sgd(neurons=128, learn_rate=0.001, momentum=0.0):
    """
    Create a model with the given layers, loss, optimizer and metrics
    loss = 'sparse_categorical_crossentropy' because it's a multi-class classification problem
    :return:
    """


    """
    Using sofmatmax activation function in the last layer because it's a multi-class classification problem.
    Using 10 units in the last layer because there are 10 classes. 
    """
    model = Sequential()
    model.add(Dense(units=neurons, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(units=10, activation='softmax'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=learn_rate, momentum=momentum)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def search_best_model():
    dataset_train = pd.read_csv('../src/fashion-mnist_train.csv')
    dataset_test = pd.read_csv('../src/fashion-mnist_test.csv')

    _model = create_model_sgd()
    network = NeuralNetwork(dataset_train, dataset_test, _model)

    model = KerasClassifier(build_fn=create_model_sgd, verbose=1)

    param_grid = {
        # 'batch_size': [16, 32, 64],
        # 'epochs': [2, 5, 10],
        # 'optimizer': ['SGD', 'Adam'],
        # Best: 0.8793166478474935 using {'batch_size': 16, 'epochs': 10, 'optimizer': 'SGD'}

        'learn_rate': [0.001, 0.01, 0.1],
        'momentum': [0.0, 0.2],
        'neurons': [64, 128, 256],
        'batch_size': [16],
        'epochs': [10],
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(network.x, network.y)

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} ({stdev}) with: {param}")


def train():
    dataset_train = pd.read_csv('../src/fashion-mnist_train.csv')
    dataset_test = pd.read_csv('../src/fashion-mnist_test.csv')

    _model = create_model_sgd()
    network = NeuralNetwork(dataset_train, dataset_test, _model)
    network.fit()

    return network

def save_model(network):
    model = network.model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    # model.summary()
    return model


if __name__ == '__main__':
    # network = train() # best accuracy: ~0.8359
    # save_model(network)

    model = load_model()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dataset_train = pd.read_csv('../src/fashion-mnist_train.csv')
    dataset_test = pd.read_csv('../src/fashion-mnist_test.csv')

    network = NeuralNetwork(dataset_train, dataset_test, model)
    network.fit()
    k=0
