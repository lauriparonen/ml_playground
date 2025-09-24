import pickle
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau
import keras
import matplotlib.pyplot as plt

data_fname = "mnist_fashion.pkl"

with open(data_fname, 'rb') as data_file:
    x_train = pickle.load(data_file)
    y_train = pickle.load(data_file)
    x_test = pickle.load(data_file)
    y_test = pickle.load(data_file)

# helper for getting training data from 28x28 -> 784x1
def flatten(x_train):
    flattened_x_train = x_train.reshape(len(x_train), -1)
    return flattened_x_train

# encode the class labsles into 1-hot vectors
def one_hot(value):
    base = np.zeros(10, dtype=int)
    np.put(base, value, 1)
    return base

def encode(y_train):
    encoded_y_train = []
    for idx, x in enumerate(y_train):
        encoded_idx = one_hot(y_train[idx])
        encoded_y_train.append(encoded_idx)
    return np.array(encoded_y_train)

y_tr_1h = encode(y_train)
flattened_x_tr = flatten(x_train) / 255.0 # normalize the data

# hyperparameters
LR = 0.02
BATCH_SIZE = 64
NUM_OF_EPOCHS = 100

# model architecture
model = Sequential()
model.add(Dense(64, input_dim=784, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
keras.optimizers.SGD(learning_rate=LR)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['mse'])
tr_hist = model.fit(flattened_x_tr, y_tr_1h, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# get predictions and accuracy for training data
y_tr_pred = np.argmax(model.predict(flattened_x_tr), axis=1)
y_true = np.argmax(y_tr_1h, axis=1)
tr_accuracy = np.mean(y_tr_pred == y_true)
print(f"Training accuracy: {tr_accuracy*100:.2f}%")

# process test data the same way as training data
flattened_x_test = flatten(x_test) / 255.0
y_test_1h = encode(y_test)

# get predictions and accuracy for test data
y_test_pred = np.argmax(model.predict(flattened_x_test), axis=1)
y_test_true = np.argmax(y_test_1h, axis=1)
test_accuracy = np.mean(y_test_pred == y_test_true)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# save test predictions to file
np.savetxt("PRED_mlp.dat", y_test_pred, fmt="%d")

# plot & save training loss curve
plt.plot(tr_hist.history['loss'], label='training loss')
plt.title('MLP on Fashion-MNIST: training loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("results/training_loss.png", dpi=150)
plt.show()
