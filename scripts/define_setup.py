"""Setup-build frames for slide 2: four PNGs that each add one element
(empty -> eyes -> +camera -> +light), sharing identical axes and view angle.

Output: ../img/define_setup_1.png ... ../img/define_setup_4.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

from lab_setup import (
    REF_BOUNDS,
    VIEW_AZIM,
    VIEW_ELEV,
    build_camera,
    build_eyes,
    build_light,
)
from pyetsimul.types import Position3D
from pyetsimul.visualization.coordinate_utils import prepare_eye_data_for_plots
from pyetsimul.visualization.setup_plots import plot_setup

OUT_DIR = Path(__file__).parent.parent / "img"

LOOK_AT = Position3D(0.0, 0.0, 0.0)


def make_axes() -> tuple[plt.Figure, plt.Axes]:
    """Create figure + 3D axes styled to match what plot_setup applies internally."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Eye Tracking Setup")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlim(*REF_BOUNDS["x"])
    ax.set_ylim(*REF_BOUNDS["y"])
    ax.set_zlim(*REF_BOUNDS["z"])
    ax.set_box_aspect([1, 1, 1])
    return fig, ax


def render_empty_step(out_path: Path) -> None:
    """Step 1: empty 3D scene (raw matplotlib; no plot_setup)."""
    fig, ax = make_axes()
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def render_step(
    out_path: Path,
    *,
    with_eyes: bool,
    with_camera: bool,
    with_light: bool,
) -> None:
    fig, ax = make_axes()

    eyes = build_eyes() if with_eyes else []
    for eye in eyes:
        eye.look_at(LOOK_AT)
    camera = build_camera(eyes) if with_camera and eyes else None
    light = build_light() if with_light else None

    look_at_per_eye = [LOOK_AT] * len(eyes)
    if eyes:
        prepared = prepare_eye_data_for_plots(
            eyes,
            look_at_per_eye,
            [light] if light else None,
            [camera] if camera else None,
        )
        eyes_data = prepared["eyes_data"]
        cr_3d_lists = prepared["cr_3d_lists"]
    else:
        eyes_data = []
        cr_3d_lists = []

    plot_setup(
        ax,
        eyes_data,
        look_at_per_eye,
        lights=[light] if light else None,
        cameras=[camera] if camera else None,
        cr_3d_lists=cr_3d_lists,
        ref_bounds=REF_BOUNDS,
    )
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_empty_step(OUT_DIR / "define_setup_1.png")
    render_step(OUT_DIR / "define_setup_2.png", with_eyes=True, with_camera=False, with_light=False)
    render_step(OUT_DIR / "define_setup_3.png", with_eyes=True, with_camera=True, with_light=False)
    render_step(OUT_DIR / "define_setup_4.png", with_eyes=True, with_camera=True, with_light=True)


if __name__ == "__main__":
    main()
