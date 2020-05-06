# Please note that most of this code was just copy pasted from an official example, since this
# is not really a part of this assignment.
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np


def get_model(datamode, data_generator, num_classes=10, input_shape=(28,28,1)):
    filepath = f"./models/basic_classifier_{datamode.name}_{num_classes}.h5"
    if os.path.exists(filepath):
        model = tensorflow.keras.models.load_model(filepath)
        # TODO: uncomment
        model.trainable = False
        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer="adam",
                      metrics=['accuracy'])
        return model
    x_train, y_train = data_generator.get_full_data_set(training=True)
    x_train = x_train.astype(np.float64)

    batch_size = 128
    epochs = 12

    x_train = x_train.astype('float32')
    x_train /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    #TODO: double check if I should use this or not.
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)

    model.save(filepath)
    model.trainable = False
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])
    return model
