import sys

from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize the dataset

x_train = x_train/255
x_test = x_test/255

# flatten the dataset to ease the model computation

x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)

# define the model

model = keras.Sequential([
    keras.layers.Dense(units=10,
                       activation='sigmoid')
    ]
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train_flatten, y_train, epochs=5)

loss_and_accuracy = model.evaluate(x_test_flatten, y_test)
print(loss_and_accuracy)
