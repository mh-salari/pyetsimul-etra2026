"""Cornea-model slides: spherical and conic eye anatomy, one image each, no title.

Outputs: ../img/cornea_spherical.png, ../img/cornea_conic.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyetsimul.core import Eye
from pyetsimul.core.cornea import Cornea, ConicCornea, SphericalCornea
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.visualization import plot_eye_anatomy

IMG_DIR = Path(__file__).parent.parent / "img"

EYE_POSITION = Position3D(0, 50, 0)
TARGET_POINT = Position3D(-20, 0, 0)
REST_ORIENTATION = RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]], validate_handedness=False)


def _remove_target_marker(ax: plt.Axes) -> None:
    """Strip plot_eye_anatomy's auto-added target scatter so the axes zoom tight on the eye."""
    for coll in list(ax.collections):
        if coll.get_label() == "Target":
            coll.remove()


def render_cornea(cornea: Cornea, out_name: str) -> None:
    """Render one eye-anatomy figure for the given cornea: no title, tight cubic view."""
    eye = Eye(cornea=cornea, pupil_boundary_points=300, eyelid_enabled=False)
    eye.set_rest_orientation(REST_ORIENTATION)
    eye.position = EYE_POSITION
    eye.look_at(TARGET_POINT)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_eye_anatomy(eye, ax=ax)
    _remove_target_marker(ax)

    # Tight cubic view around the eye centre (auto-scaling otherwise drags the
    # range to the world origin and flattens the eye).
    half = eye.axial_length / 2 * 1.1
    cx, cy, cz = EYE_POSITION.x, EYE_POSITION.y, EYE_POSITION.z
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("")
    if ax.get_legend() is not None:
        ax.get_legend().set_visible(False)

    out_path = IMG_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    """Render both cornea-model figures."""
    render_cornea(SphericalCornea(), "cornea_spherical.png")
    render_cornea(ConicCornea(), "cornea_conic.png")


if __name__ == "__main__":
    main()
