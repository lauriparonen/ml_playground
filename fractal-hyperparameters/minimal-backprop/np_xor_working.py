import numpy as np

# === activations ===
def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - np.tanh(x)**2

# === loss ===
def mse(y_pred, y_true): return np.mean((y_pred - y_true)**2)
def dmse(y_pred, y_true): return 2 * (y_pred - y_true) / len(y_true)

# === data ===
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# === init ===
def init_params(scale=0.5):
    rng = np.random.default_rng()
    W1 = rng.normal(0, 1, (2, 4)) * scale
    b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 1, (4, 1)) * scale
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# === one training step ===
def train_step(X, Y, W1, b1, W2, b2, lr):
    # forward
    Z1 = X @ W1 + b1
    A1 = tanh(Z1)
    Z2 = A1 @ W2 + b2
    A2 = tanh(Z2)
    loss = mse(A2, Y)

    # backward
    dZ2 = dmse(A2, Y) * dtanh(Z2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * dtanh(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    return loss, W1, b1, W2, b2

# === train loop ===
def train(X=None, Y=None, lr=0.1, steps=500, verbose=False, init_scale=0.5):
    if X is None:
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
    if Y is None:
        Y = np.array([[0],[1],[1],[0]])
    W1, b1, W2, b2 = init_params(init_scale)

    for i in range(steps):
        loss, W1, b1, W2, b2 = train_step(X, Y, W1, b1, W2, b2, lr)
        if verbose and i % 50 == 0:
            print(f"step {i:3d} | loss: {loss:.6f}")
    return W1, b1, W2, b2, loss

# === test run ===
if __name__ == "__main__":
    W1, b1, W2, b2, loss = train(X, Y, lr=0.1, steps=2000, verbose=True)
    print("\nfinal loss:", loss)
    out = tanh(tanh(X @ W1 + b1) @ W2 + b2)
    print("predictions:\n", out.round(3))