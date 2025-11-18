# heightmapvisual.py — visualize all saved heightmaps (2D + 3D)
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


HEIGHTMAP_DIR = "heightmaps"


def plot_heightmap_2d(X, Y, DZ, out_path):
    plt.figure(figsize=(7, 6))
    plt.title("Heightmap Δz(x,y) — 2D Heatmap")
    im = plt.imshow(
        DZ,
        origin="lower",
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        cmap="viridis",
        aspect="equal"
    )
    plt.colorbar(im, label="Δz [mm]")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print("  Saved 2D heatmap →", out_path)


def plot_heightmap_3d(X, Y, DZ, out_path, elev=35, azim=135):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, DZ,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rstride=2,
        cstride=2,
    )

    ax.set_title("Heightmap Δz(x,y) — 3D Surface")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Δz [mm]")

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print("  Saved 3D surface →", out_path)

    # Keep figure open so you can scroll/rotate with the mouse if running interactively
    # Comment the next line out if you don't want GUI windows:
    plt.show()

    plt.close()


def process_all_heightmaps():
    if not os.path.isdir(HEIGHTMAP_DIR):
        print("No 'heightmaps/' directory found.")
        return

    files = [f for f in os.listdir(HEIGHTMAP_DIR) if f.endswith("_heightmap.npz")]
    if not files:
        print("No *_heightmap.npz files found in 'heightmaps/'.")
        return

    for fname in files:
        path = os.path.join(HEIGHTMAP_DIR, fname)
        print("\nProcessing:", path)

        data = np.load(path)
        X = data["X"]
        Y = data["Y"]
        DZ = data["DZ"]

        base = fname.replace(".npz", "")
        out_2d = os.path.join(HEIGHTMAP_DIR, base + "_2D.png")
        out_3d = os.path.join(HEIGHTMAP_DIR, base + "_3D.png")

        plot_heightmap_2d(X, Y, DZ, out_2d)
        plot_heightmap_3d(X, Y, DZ, out_3d)


if __name__ == "__main__":
    process_all_heightmaps()
