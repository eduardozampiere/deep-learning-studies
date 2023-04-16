from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import random as rn
import pandas as pd

def get_data():
    orginal_data = pd.read_csv('../src/cardio.csv', sep=';')
    # remove the first column because it is the id of the patient
    data = orginal_data.iloc[:, 1:]

    # get the features
    x = data.iloc[:, :-1].values

    # get the labels
    y = data.iloc[:, -1:].values

    # split the data into trainning and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    # scale the data
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test, y_train, y_test


def one_layer_model(x_train, x_test, y_train, y_test, epochs=10):
    ppn = Perceptron(max_iter=epochs, eta0=0.1)
    ppn.fit(x_train, y_train.ravel())


    # predict the test data
    y_pred = ppn.predict(x_test)

    # get the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # get the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return cm, accuracy


def multiple_layers_model(x_train, x_test, y_train, y_test, epochs=10):
    SEED = 0
    np.random.seed(SEED)
    rn.seed(SEED)
    tf.random.set_seed(SEED)

    model = Sequential()
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=256, epochs=epochs)

    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cm, accuracy

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()
    epochs = 1000
    # train the model
    cm1, accuracy1 = one_layer_model(x_train, x_test, y_train, y_test, epochs=epochs)
    cm2, accuracy2 = multiple_layers_model(x_train, x_test, y_train, y_test, epochs=epochs)

    # print('One layer model confusion matrix: ', cm1)
    print('\nOne layer model accuracy: ', accuracy1)
    print('----------------------------------')
    # print('Multiple layers model confusion matrix: ', cm2)
    print('Multiple layers model accuracy: ', accuracy2)
    k = 0