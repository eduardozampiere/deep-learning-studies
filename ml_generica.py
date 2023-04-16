import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

dataset = keras.datasets.mnist.load_data()
(train_x, train_y), (test_x, test_y) = dataset

# Normalize the data
# 0 - 255 -> 0 - 1
train_x = train_x / 255
test_x = test_x / 255


# Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(20, activation='relu'),
    # keras.layers.Dense(50, activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
# epochs: number of iterations
model.fit(train_x, train_y, epochs=5)

# Evaluate the model
loss, accuracy = model.evaluate(test_x, test_y)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

y_predicted = model.predict(test_x)

y_predicted_labels = [np.argmax(i) for i in y_predicted]

k = tf.math.confusion_matrix(labels=test_y, predictions=y_predicted_labels)

plt.figure(figsize=(10,7))
sn.heatmap(k, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

