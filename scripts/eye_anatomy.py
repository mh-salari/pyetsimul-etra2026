"""Eye-anatomy figure for slide 01b: the conic-cornea "right eye" from
examples/eye_anatomy.py (eyelid 75% open) rendered with the simulator's own
anatomy visualization, cleaned up for a slide (no legend, no title, no axes).

Output: ../img/eye_anatomy.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyetsimul.core import Eye
from pyetsimul.core.cornea import ConicCornea
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_eye_anatomy

OUT_PATH = Path(__file__).parent.parent / "img" / "eye_anatomy.png"

TARGET_POINT = Position3D(10, 10, -10)
EYELID_OPENNESS = 0.75


def main() -> None:
    eye = Eye(cornea=ConicCornea(), eyelid_enabled=True)
    eye.eyelid.openness = EYELID_OPENNESS
    eye.set_rest_orientation_at_target(TARGET_POINT)
    eye.look_at(TARGET_POINT)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_eye_anatomy(eye, ax=ax)

    # Strip the slide-cluttering chrome plot_eye_anatomy adds.
    ax.set_title("")
    for text in list(ax.texts):  # removes the "Eye openness: X%" overlay
        text.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    # No custom view_init: keep the default 3D angle, matching examples/eye_anatomy.py.

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
