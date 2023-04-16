from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron


class DeepLearning:
    def __init__(self, data, labels, model):
        self.data = data
        self.labels = labels
        self.model = model

    def train(self):
        self.model.fit(self.data, self.labels)

    def score(self):
        return self.model.score(self.data, self.labels)


if __name__ == '__main__':
    data_train, data_labels = load_digits(return_X_y=True)
    _linear_model = Perceptron()
    model = DeepLearning(data_train, data_labels, _linear_model)
    model.train()
    print(model.score())
