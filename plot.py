"""Chart of the result, for the README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "standard": ("#888888", "--", "o"),
    "GNS known": ("#1f77b4", "-", "o"),
    "GNS kernel": ("#d62728", "-", "o"),
    "pooled mean": ("#cccccc", ":", None),
}


def draw(results, dimensions, path="imse.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, (colour, dash, marker) in STYLE.items():
        ax.plot(
            dimensions,
            [results[d][name] for d in dimensions],
            label=name,
            color=colour,
            linestyle=dash,
            marker=marker,
            linewidth=2,
        )

    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dimensions)
    ax.set_xticklabels(dimensions)
    ax.set_xlabel("conditioning dimensions")
    ax.set_ylabel("IMSE vs closed form (log scale)")
    ax.set_title("Green nested simulation loses its advantage as dimension grows")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path
