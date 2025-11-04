from joblib import Parallel, delayed
import numpy as np
from np_xor_working import train

# === run one point in the grid ===
def run_point(lr, scale, steps=200, trials=1):
    losses = []
    for _ in range(trials):
        try:
            _, _, _, _, loss = train(lr=lr, steps=steps, verbose=False, init_scale=scale)
        except Exception as e:
            print(f"error at lr={lr}, scale={scale}: {e}")
            loss = np.nan
        losses.append(loss)
    return np.mean(losses)

# === run full grid sweep ===
def run_grid(lr_vals, scale_vals, steps=200, trials=1, n_jobs=-1):
    print(f"running grid {len(lr_vals)}×{len(scale_vals)}...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_point)(lr, scale, steps, trials)
        for lr in lr_vals
        for scale in scale_vals
    )
    return np.array(results).reshape(len(lr_vals), len(scale_vals))

# === main runner ===
if __name__ == "__main__":
    lr_vals = np.geomspace(1e-3, 1.0, 300)
    scale_vals = np.geomspace(1e-2, 1.0, 300)

    grid = run_grid(lr_vals, scale_vals, steps=200, trials=1)
    np.save("results/loss_grid.npy", grid)
    print("saved to results/loss_grid.npy")