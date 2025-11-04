import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.callbacks import EarlyStopping
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data
X, y = make_moons(n_samples=2000, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Hyperparam ranges
lrs = np.logspace(-4, 0, 50)  # 1e-4 to 1.0, 50 points
batch_sizes = np.logspace(0, np.log10(512), 50).astype(int)  # 1 to 512, log-spaced
EPOCHS = 5

# Initialize results array
results = np.zeros((len(batch_sizes), len(lrs)))

# Load checkpoint if exists
prefix = "moons_lr1e-4to1.0x50_batch1to512x50_epochs5"
results_file = f"results/{prefix}_results.npy"
if os.path.exists(results_file):
    results = np.load(results_file)
    logging.info(f"Loaded checkpoint from {results_file}")

# Early stopping callback
early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=2, mode='max', restore_best_weights=True)

# Training function
def train_model(batch_size, lr, i, j):
    try:
        logging.info(f"Training: batch_size={batch_size}, lr={lr:.5f}")
        model = keras.Sequential([
            keras.Input(shape=(2,)),
            layers.Dense(32, activation="sigmoid"),
            layers.Dense(1, activation="sigmoid")
        ])
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.0)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=EPOCHS,
            verbose=0,
            callbacks=[early_stop]
        )
        acc = history.history["val_accuracy"][-1]
        K.clear_session()
        gc.collect()
        return acc
    except Exception as e:
        logging.error(f"Error at batch_size={batch_size}, lr={lr:.5f}: {str(e)}")
        return 0.0  # Return 0 accuracy on failure

# Grid search
def run_grid_search():
    for i, batch_size in enumerate(tqdm(batch_sizes, desc="Batch sizes")):
        for j, lr in enumerate(lrs):
            if results[i, j] == 0.0:  # Skip if already computed (from checkpoint)
                results[i, j] = train_model(batch_size, lr, i, j)
            if (i * len(lrs) + j) % 50 == 0:
                os.makedirs("results", exist_ok=True)
                np.save(f"results/{prefix}_results.npy", results)
                np.save(f"results/{prefix}_lrs.npy", lrs)
                np.save(f"results/{prefix}_batch_sizes.npy", batch_sizes)
                logging.info("Checkpoint saved")

# Run and save
if __name__ == '__main__':
    try:
        run_grid_search()
    except KeyboardInterrupt:
        logging.warning("Script interrupted, saving checkpoint")
    finally:
        os.makedirs("results", exist_ok=True)
        np.save(f"results/{prefix}_results.npy", results)
        np.save(f"results/{prefix}_lrs.npy", lrs)
        np.save(f"results/{prefix}_batch_sizes.npy", batch_sizes)
        logging.info(f"Done, results saved to ./results/{prefix}_*.npy")

        # Plot results
        plt.figure(figsize=(10, 8))
        plt.imshow(results, extent=[np.log10(lrs.min()), np.log10(lrs.max()), batch_sizes.min(), batch_sizes.max()],
                   origin='lower', cmap='binary', interpolation='none', aspect='auto')
        plt.colorbar(label='Validation Accuracy')
        plt.xlabel('Log10(Learning Rate)')
        plt.ylabel('Batch Size')
        plt.title('Trainability Landscape (Accuracy)')
        plt.contour(np.log10(lrs), batch_sizes, results, levels=[0.8], colors='red')  # Boundary at acc=0.8
        plt.savefig(f"results/{prefix}_plot.png")
        plt.close()
        logging.info(f"Plot saved to results/{prefix}_plot.png")