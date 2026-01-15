import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def plot_analysis(mean_grid, var_grid, class_map, extent):
    """
    Create a 3-panel figure:
      1) Mean map
      2) Variance map
      3) Quadrant-based classification map

    Parameters
    ----------
    mean_grid : np.ndarray
        2D array with per-cell mean values.
    var_grid : np.ndarray
        2D array with per-cell variance values.
    class_map : np.ndarray
        2D int array with values in {0,1,2,3} and -1 for "no data".
    extent : list[float]
        [xmin, xmax, ymin, ymax] passed to imshow so axes match the simulation domain.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure (caller can plt.show() or fig.savefig()).
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 1) Mean map
    im1 = axs[0].imshow(mean_grid, origin="lower", cmap="viridis", extent=extent)
    axs[0].set_title("Mean (accumulated metric)")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    fig.colorbar(im1, ax=axs[0])

    # 2) Variance map
    im2 = axs[1].imshow(var_grid, origin="lower", cmap="magma", extent=extent)
    axs[1].set_title("Variance (instability)")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    fig.colorbar(im2, ax=axs[1])

    # 3) Classification map
    # Mask cells where we have no data (-1) so they render as the background color.
    class_map_masked = np.ma.masked_where(class_map == -1, class_map)

    # Fixed colors for classes 0..3 (matches the legend below)
    cmap = ListedColormap(["green", "gold", "red", "blue"])

    im3 = axs[2].imshow(
        class_map_masked,
        origin="lower",
        cmap=cmap,
        extent=extent,
        interpolation="nearest",
    )
    axs[2].set_title("Dynamic classification")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    # Background for masked areas (no data)
    axs[2].set_facecolor("black")

    # Custom legend
    labels = [
        "High mean / Low variance",
        "High mean / High variance",
        "Low mean / High variance",
        "Low mean / Low variance",
    ]
    colors = ["green", "gold", "red", "blue"]
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(4)]
    axs[2].legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig
