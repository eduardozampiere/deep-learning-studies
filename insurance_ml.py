import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression


def sigmoid(x):
    for item in x:
        yield 1 / (1 + np.exp(-item))


def create_database():
    np.random.seed(42)
    ages = np.random.randint(15, high=70, size=40)

    labels = [int(age >= 30) for age in ages]


    for i in range(0, 3):
        rand = np.random.randint(0, len(ages) - 1)
        labels[rand] = int(not bool(labels[rand]))


    return ages, labels


def linear_regression(ages, labels):
    model = LinearRegression()
    model.fit(ages.reshape(-1, 1), labels)
    m = model.coef_[0]
    b = model.intercept_

    limit = (0.5 - b) / m

    plt.plot(ages, m * ages + b, color='blue')
    plt.plot([limit, limit], [0, 0.5], '--', color='green')
    plt.scatter(ages, labels, color='red')
    plt.show()


def logistic_regression(ages, labels):
    model = LogisticRegression()
    model.fit(ages.reshape(-1, 1), labels)

    m = model.coef_[0]
    b = model.intercept_

    limit = abs(b)/abs(m)

    x = np.arange(0, 70, 0.1)
    y = list(sigmoid(m*x + b))

    plt.plot(x, y, color='blue')
    plt.plot([limit, limit], [0, 0.5], '--', color='green')
    plt.scatter(ages, labels, color='red')
    plt.show()



if __name__ == '__main__':
    ages, labels = create_database()
    # linear_regression(ages, labels)
    logistic_regression(ages, labels)

