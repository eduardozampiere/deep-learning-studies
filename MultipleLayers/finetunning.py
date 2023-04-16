import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from stroke import NeuralNetwork
# Deprecated
from keras.wrappers.scikit_learn import KerasClassifier


def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(units=6, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(units=6, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    dataset_original = pd.read_csv('../src/stroke.csv')

    _model = NeuralNetwork(dataset_original, optimizer='sgd')
    _model.init_treatment()

    x = _model.x
    y = _model.y
    x_test = _model.x_test
    y_test = _model.y_test

    model = KerasClassifier(build_fn=create_model, verbose=2)
    optimizer = ['SGD', 'Adam']
    batch_size = [16, 32, 64]
    epochs = [10 * i for i in range(1, 6)]

    param_grid = {
        'optimizer': optimizer,
        'batch_size': batch_size,
        'epochs': epochs
    }

    # cv means cross validation (how many times the model will be trained for each combination of parameters)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
    grid_result = grid.fit(x, y)

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean} ({stdev}) with: {param}")

    """
    Best: 0.9584914326667786 using {'batch_size': 16, 'epochs': 40, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 16, 'epochs': 10, 'optimizer': 'SGD'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 16, 'epochs': 10, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 16, 'epochs': 20, 'optimizer': 'SGD'}
        0.9574723362922668 (0.004610642494248662) with: {'batch_size': 16, 'epochs': 20, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 16, 'epochs': 30, 'optimizer': 'SGD'}
        0.9567079901695251 (0.004979364878052535) with: {'batch_size': 16, 'epochs': 30, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 16, 'epochs': 40, 'optimizer': 'SGD'}
        0.9584914326667786 (0.004315166229429812) with: {'batch_size': 16, 'epochs': 40, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 16, 'epochs': 50, 'optimizer': 'SGD'}
        0.9569630980491638 (0.005495965658324884) with: {'batch_size': 16, 'epochs': 50, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 32, 'epochs': 10, 'optimizer': 'SGD'}
        0.9572175502777099 (0.00488515366256753) with: {'batch_size': 32, 'epochs': 10, 'optimizer': 'Adam'}
        0.9572182059288025 (0.00500994568974209) with: {'batch_size': 32, 'epochs': 20, 'optimizer': 'SGD'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 32, 'epochs': 20, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 32, 'epochs': 30, 'optimizer': 'SGD'}
        0.9579812526702881 (0.005533200772627719) with: {'batch_size': 32, 'epochs': 30, 'optimizer': 'Adam'}
        0.9574723362922668 (0.004610642494248662) with: {'batch_size': 32, 'epochs': 40, 'optimizer': 'SGD'}
        0.9582366585731507 (0.0043012876077849795) with: {'batch_size': 32, 'epochs': 40, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 32, 'epochs': 50, 'optimizer': 'SGD'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 32, 'epochs': 50, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 64, 'epochs': 10, 'optimizer': 'SGD'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 64, 'epochs': 10, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 64, 'epochs': 20, 'optimizer': 'SGD'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 64, 'epochs': 20, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 64, 'epochs': 30, 'optimizer': 'SGD'}
        0.9572175502777099 (0.004681602372258911) with: {'batch_size': 64, 'epochs': 30, 'optimizer': 'Adam'}
        0.957217562198639 (0.00461175231171542) with: {'batch_size': 64, 'epochs': 40, 'optimizer': 'SGD'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 64, 'epochs': 40, 'optimizer': 'Adam'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 64, 'epochs': 50, 'optimizer': 'SGD'}
        0.9577271103858948 (0.0046655190220598166) with: {'batch_size': 64, 'epochs': 50, 'optimizer': 'Adam'}
    """