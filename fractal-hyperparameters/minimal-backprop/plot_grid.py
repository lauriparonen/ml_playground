import numpy as np
import plotly.express as px
import argparse

def plot_loss_grid(path: str,
                   engine: str = "plotly",
                   title: str = "Trainability landscape",
                   threshold: float = 0.05,
                   binary: bool = True,
                   zcap: float = 1.0):
    
    Z = np.load(path)
    lr_vals = np.geomspace(1e-3, 1.0, Z.shape[0])
    scale_vals = np.geomspace(1e-2, 1.0, Z.shape[1])

    if binary:
        # trainability mask: 1 = trained (loss < threshold), 0 = fail
        Z = (Z < threshold).astype(float)

    else:
        # cap extreme loss values for clarity
        Z = np.clip(Z, 0, zcap)

    if engine == "matplotlib":
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 7))
        im = plt.imshow(Z, origin='lower', aspect='auto',
                        extent=[scale_vals[0], scale_vals[-1], lr_vals[0], lr_vals[-1]],
                        cmap='gray' if binary else 'viridis')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Init scale")
        plt.ylabel("Learning rate")
        plt.colorbar(im, label="Trainable" if binary else "Loss")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    elif engine == "plotly":
        fig = px.imshow(Z,
                        x=scale_vals, y=lr_vals,
                        labels={"x": "Init scale", "y": "Learning rate",
                                "color": "Trainable" if binary else "Loss"},
                        color_continuous_scale=["black", "white"] if binary else "viridis",
                        origin="lower",
                        aspect="auto")
        fig.update_layout(title=title,
                          xaxis_type="log", yaxis_type="log")
        fig.write_html("plot.html", auto_open=True)
    else:
        raise ValueError(f"unknown engine: {engine}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", type=str, default="results/loss_grid.npy")
    parser.add_argument("--engine", type=str, default="plotly", choices=["plotly", "matplotlib"])
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Threshold for loss to count as trainable (used in binary mode)")
    parser.add_argument("--binary", action="store_true", default=False,
                        help="Plot binary trainability mask instead of raw loss")
    parser.add_argument("--zcap", type=float, default=1.0,
                        help="Cap max value of loss in heatmap (non-binary only)")
    args = parser.parse_args()

    plot_loss_grid(path=args.npy,
                   engine=args.engine,
                   title="Trainability landscape",
                   threshold=args.threshold,
                   binary=args.binary,
                   zcap=args.zcap)