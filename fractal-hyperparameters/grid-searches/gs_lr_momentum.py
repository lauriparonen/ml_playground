import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)

X, y = make_moons(n_samples=2000, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# hyperparam ranges
lrs = np.logspace(-4, 0, 50)         # 1e-4 to 1.0, 50 points
momenta = np.linspace(0.0, 0.99, 50)
EPOCHS = 5

results = np.zeros((len(momenta), len(lrs)))

# grid search loop
for i, mom in enumerate(momenta):
    for j, lr in enumerate(lrs):
        if (i * len(lrs) + j) % 50 == 0:  
            os.makedirs("results", exist_ok=True)
            np.save("results/results.npy", results)
            np.save("results/lrs.npy", lrs)
            np.save("results/momenta.npy", momenta)
            print("checkpoint saved")

        print(f"momentum={mom:.2f}, lr={lr:.5f}")

        model = keras.Sequential([
            keras.Input(shape=(2,)),
            layers.Dense(32, activation="sigmoid"),
            layers.Dense(1, activation="sigmoid")
        ])

        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=mom)
        model.compile(optimizer=optimizer,
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=32,
            epochs=EPOCHS,
            verbose=0
        )

        acc = history.history["val_accuracy"][-1]
        results[i, j] = acc

        K.clear_session()

prefix = "moons_lr1e-4to1.0x50_mom0.0to0.99x50_epochs5"

# final save
os.makedirs("results", exist_ok=True)
np.save(f"results/{prefix}_results.npy", results)
np.save(f"results/{prefix}_lrs.npy", lrs)
np.save(f"results/{prefix}_momenta.npy", momenta)

print(f"done, results saved to ./results/{prefix}_*.npy")