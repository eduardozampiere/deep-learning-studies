"""
Download the data from Kaggle: https://www.kaggle.com/competitions/dogs-vs-cats/data
"""
import os
import glob

from keras.callbacks import ModelCheckpoint
from keras.utils import image_dataset_from_directory
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, RandomFlip, RandomRotation, RandomZoom, \
    Dropout, BatchNormalization
from keras.models import Sequential
import tensorflow as tf
from keras.models import load_model



def data_aumentation():
    pass


def create_model():
    # augmentation data layer
    model = Sequential([
        Rescaling(1./255),
        RandomFlip("horizontal"),
        RandomRotation(0.15),
        RandomZoom(0.2),
    ])


    # Convolutional layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


if __name__ == '__main__':
    train_dataset = image_dataset_from_directory("../src/dog-cat/train", image_size=(180, 180), batch_size=32)
    validation_dataset = image_dataset_from_directory("../src/dog-cat/validation", image_size=(180, 180), batch_size=32)

    model = create_model()
    # model = load_model("model.keras")

    callbacks = [
        ModelCheckpoint(filepath="model_1.keras", save_best_only=True),
    ]

    model.fit(train_dataset, epochs=2, callbacks=callbacks, validation_data=validation_dataset)
