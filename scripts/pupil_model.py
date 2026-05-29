"""Pupil-model slide: realistic (Wyatt 1995) vs circular pupil outline.

Boundary points are in the pupil's local frame, so the figure is independent of
eye position and orientation.

Output: ../img/pupil_model.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyetsimul.core import Eye
from pyetsimul.types import Direction3D

OUT_PATH = Path(__file__).parent.parent / "img" / "pupil_model.png"

DIAMETER_MM = 6.0


def _closed_boundary_xy(eye: Eye) -> tuple[np.ndarray, np.ndarray]:
    """Return the pupil boundary x/y (mm) as a closed polygon."""
    boundary = eye.pupil.get_boundary_points()
    x = np.append(boundary[0, :], boundary[0, 0])
    y = np.append(boundary[1, :], boundary[1, 0])
    return x, y


def main() -> None:
    """Render the realistic-vs-circular pupil shape figure."""
    eye_realistic = Eye(pupil_type="realistic", pupil_boundary_points=100)
    eye_circular = Eye(pupil_type="elliptical", pupil_boundary_points=100)

    eye_realistic.pupil.set_diameter(DIAMETER_MM)
    radius = DIAMETER_MM / 2
    eye_circular.pupil.x_pupil = Direction3D(radius, 0, 0)
    eye_circular.pupil.y_pupil = Direction3D(0, radius, 0)

    circular_x, circular_y = _closed_boundary_xy(eye_circular)
    realistic_x, realistic_y = _closed_boundary_xy(eye_realistic)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(circular_x, circular_y, "b-", linewidth=2, alpha=0.7, label="Circular pupil")
    ax.plot(realistic_x, realistic_y, "r-", linewidth=2, label="Realistic pupil")

    center = eye_realistic.pupil.pos_pupil
    ax.scatter(center.x, center.y, color="black", s=50, marker="+", linewidth=2, label="Pupil center")

    lim = radius * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
