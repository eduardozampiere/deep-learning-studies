import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense


class NeuralNetwork:
    def __init__(self, dataset: pd.DataFrame, test_size: float = 0.2,
                 layers=None, loss='binary_crossentropy', optimizer='adam', metrics=None):
        self.dataset = dataset
        assert self.dataset.empty is False, "Dataset must not be empty"

        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()
        self.column_transformer = ColumnTransformer([], remainder='passthrough')
        self.standard_scaler = StandardScaler()
        self.model = None

        self.x = None
        self.y = None

        self.x_test = None
        self.y_test = None
        self.test_size = test_size

        assert loss, "Loss must not be empty"
        assert optimizer, "Optimizer must not be empty"
        self.init_model(layers, metrics, loss, optimizer)

    def init_model(self, layers=None, metrics=None, loss='binary_crossentropy', optimizer='adam'):
        """
        Initialize the model with the given layers, loss, optimizer and metrics
        :param layers:
        :param loss:
        :param optimizer:
        :param metrics:
        :return:
        """
        self.model = Sequential()
        if not layers:
            layers = [
                Dense(units=6, activation='relu'),
                Dense(units=6, activation='relu'),
                Dense(units=1, activation='sigmoid')
            ]

        if not metrics:
            metrics = ['accuracy']

        for layer in layers:
            self.model.add(layer)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model

    def init_treatment(self):
        """
        Initialize the treatment of the dataset to be used in the model
        :return:
        """
        self.remove_nulls()
        self.get_x_y()
        self.encode_categorical()
        self.get_test_data()
        self.normalize_data()


    def remove_nulls(self):
        # show nulls in dataset
        # null_rows = dataset[dataset.isnull().any(axis=1)]

        # Remove nulls
        self.dataset = self.dataset.dropna()

        null_rows = self.dataset[self.dataset.isnull().any(axis=1)]
        assert null_rows.shape[0] == 0, "There are still null rows"


    def encode_categorical(self):
        """
        encode categorical data to numerical data
        [gender, ever_married, work_type, Residence_type, smoking_status]
        :return:
        """
        binary_columns = ['ever_married', 'Residence_type']
        not_binary_columns = ['gender', 'work_type', 'smoking_status']

        self.column_transformer.transformers = [('encoder', self.one_hot_encoder, not_binary_columns)]


        self.x[binary_columns] = self.x[binary_columns].apply(self.label_encoder.fit_transform)

        # transform the dataset applying the column transformer with one hot encoder
        self.x = self.column_transformer.fit_transform(self.x)


    def get_x_y(self):
        """
        get x and y from dataset
        :return:
        """
        self.x = self.dataset.iloc[:, 1:-1]
        self.y = self.dataset.iloc[:, -1]

        assert self.x.shape[0] == self.y.shape[0], "X and Y must have the same number of rows"


    def get_test_data(self):
        """
        get test data from x and y
        :return:
        """
        self.x, self.x_test, self.y, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size, random_state=0)

        assert self.x.shape[0] == self.y.shape[0], "X and Y must have the same number of rows"
        assert self.x_test.shape[0] == self.y_test.shape[0], "X test and Y test must have the same number of rows"


    def normalize_data(self):
        """
        normalize scale of data
        :return:
        """
        self.x = self.standard_scaler.fit_transform(self.x)
        self.x_test = self.standard_scaler.transform(self.x_test)


    def fit(self, batch_size=16, epochs=100):
        self.model.fit(self.x, self.y, batch_size=batch_size, epochs=epochs, verbose=1)


    def evaluate(self):
       return self.model.evaluate(self.x_test, self.y_test)

    def predict(self):
        y_pred = self.model.predict(self.x_test)
        y_pred = (y_pred > 0.5)

        accuracy = accuracy_score(self.y_test, y_pred)
        return y_pred, accuracy


if __name__ == '__main__':
    dataset_original = pd.read_csv('../src/stroke.csv')

    # sgd ou adam ? Testar
    model = NeuralNetwork(dataset_original, optimizer='sgd')
    model.init_treatment()
    model.fit(epochs=5)

    # _, accuracy = model.predict()
    loss, accuracy = model.evaluate()
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")


    k = 0
