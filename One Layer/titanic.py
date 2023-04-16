from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd


class DeepLearningModel:
    def __init__(self, data, test, model):
        super().__init__()
        self.data = data
        self.labels = []

        self.model = model

        self.test = test
        self.test_labels = []

    def clear_data(self):

        self.data = self.data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        self.test = self.test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

        self.test = self.test.dropna()
        self.data = self.data.dropna()

        self.encode_text()

        self.test = self.test.dropna()
        self.data = self.data.dropna()

        self.labels = self.data['Survived']
        self.data = self.data.drop(['Survived'], axis=1)


    def encode_text(self):
        label_encoder = LabelEncoder()
        self.data['Sex'] = label_encoder.fit_transform(self.data['Sex'])
        self.test['Sex'] = label_encoder.fit_transform(self.test['Sex'])

        hot_encoder = OneHotEncoder(handle_unknown='ignore')
        embarked_data = pd.DataFrame(hot_encoder.fit_transform(self.data[['Embarked']]).toarray())
        self.data = self.data.iloc[:, :-1]
        self.data = pd.concat([self.data, embarked_data], axis=1, ignore_index=False)

        embarked_test = pd.DataFrame(hot_encoder.fit_transform(self.test[['Embarked']]).toarray())
        self.test = self.test.iloc[:, :-1]
        self.test = pd.concat([self.test, embarked_test], axis=1, ignore_index=False)

    def normalize_data(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data.values)
        self.test = scaler.fit_transform(self.test.values)


    def train(self):
        self.model.fit(self.data, self.labels)

    def predict(self):
        self.test_labels = self.model.predict(self.data)

        k = confusion_matrix(self.labels, self.test_labels)

        k=0


if __name__== '__main__':
    train = pd.read_csv('../src/titanic_train.csv')
    test = pd.read_csv('../src/titanic_test.csv')




    _perceptron = Perceptron(max_iter=1000, eta0=0.1)

    model = DeepLearningModel(train, test, _perceptron)
    model.clear_data()
    model.normalize_data()
    model.train()
    model.predict()


    k = 0



