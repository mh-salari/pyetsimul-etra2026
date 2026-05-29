"""Eyelid slide: 3D eye anatomy at three eyelid-openness levels (50 %, 75 %, 100 %).

Output: ../img/10_eyelid.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyetsimul.core import Eye
from pyetsimul.core.cornea import SphericalCornea
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_eye_anatomy

OUT_PATH = Path(__file__).parent.parent / "img" / "10_eyelid.png"

TARGET_POINT = Position3D(10, 10, -10)
OPENNESS_LEVELS = [(0.50, "50%"), (0.75, "75%"), (1.00, "100%")]


def main() -> None:
    """Render the eyelid openness comparison figure."""
    eyes = []
    for openness, _ in OPENNESS_LEVELS:
        eye = Eye(cornea=SphericalCornea(), eyelid_enabled=True)
        eye.eyelid.openness = openness
        eye.set_rest_orientation_at_target(TARGET_POINT)
        eye.look_at(TARGET_POINT)
        eyes.append(eye)

    fig = plt.figure(figsize=(18, 6))
    axes = []
    for i, (eye, (_, label)) in enumerate(zip(eyes, OPENNESS_LEVELS, strict=False)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        plot_eye_anatomy(eye, ax=ax)
        ax.set_title(f"{label} Eyelid Openness", fontsize=14, fontweight="bold")
        # plot_eye_anatomy stamps an "X% open" caption on the axes; strip it.
        for text in list(ax.texts):
            text.remove()
        axes.append(ax)

    # Match axis limits across all three so geometry is comparable.
    xlim = axes[0].get_xlim()
    ylim = axes[0].get_ylim()
    zlim = axes[0].get_zlim()
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(elev=-30, azim=45, roll=60)
        if ax.get_legend() is not None:
            ax.get_legend().set_visible(False)

    plt.tight_layout(pad=2.0)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
