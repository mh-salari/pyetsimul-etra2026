"""Target-sweep animation: the gaze cycles through the HV9 centre + 4 cardinal
midpoints, with the 3D scene (left) and camera view (right) side by side.

Output: ../img/pupil_glint.mp4
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_setup_and_camera_view

from lab_setup import (
    CAL_HALF_H,
    CAL_HALF_W,
    REF_BOUNDS,
    VIEW_AZIM,
    VIEW_ELEV,
    build_camera,
    build_eyes,
    build_light,
)

OUT_PATH = Path(__file__).parent.parent / "img" / "pupil_glint.mp4"

# Cardinal midpoints of the calibration area on the screen plane (y = 0).
TARGETS: list[Position3D] = [
    Position3D(0.0, 0.0, 0.0),
    Position3D(0.0, 0.0, CAL_HALF_H),
    Position3D(CAL_HALF_W, 0.0, 0.0),
    Position3D(0.0, 0.0, -CAL_HALF_H),
    Position3D(-CAL_HALF_W, 0.0, 0.0),
]

# Padding around the feature-bounding-box used to crop the camera view.
CAMERA_CROP_MARGIN = 0.2

FPS = 30
SECONDS_PER_TARGET = 1.5
FRAMES_PER_TARGET = int(SECONDS_PER_TARGET * FPS)
TOTAL_FRAMES = len(TARGETS) * FRAMES_PER_TARGET

PROGRESS_BG = "#eeeeee"
PROGRESS_FG = "#999999"


def _compute_feature_bounds(
    eyes: list[Eye],
    camera: Camera,
    light: Light,
    targets: list[Position3D],
) -> tuple[float, float, float, float]:
    """Find the bounding box of all 2D features across the full target sweep."""
    xs: list[float] = []
    ys: list[float] = []
    for target in targets:
        for eye in eyes:
            eye.look_at(target)
            image = camera.take_image(eye, [light])
            for pt in [image.pupil_center] + (image.pupil_boundary or []) + list(image.corneal_reflections):
                if pt is not None:
                    xs.append(pt.x)
                    ys.append(pt.y)
    return min(xs), max(xs), min(ys), max(ys)


def _apply_crop(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Set the camera-view limits; Y flipped so 0 is at the bottom."""
    cur_xlim = ax.get_xlim()
    ax.set_xlim(xlim if cur_xlim[0] < cur_xlim[1] else (xlim[1], xlim[0]))
    ax.set_ylim(min(ylim), max(ylim))  # Y flipped


def main() -> None:
    """Render the target sweep MP4."""
    eyes = build_eyes()
    camera = build_camera(eyes)
    light = build_light()

    xmin, xmax, ymin, ymax = _compute_feature_bounds(eyes, camera, light, TARGETS)
    x_margin = (xmax - xmin) * CAMERA_CROP_MARGIN
    y_margin = (ymax - ymin) * CAMERA_CROP_MARGIN
    crop_xlim = (xmin - x_margin, xmax + x_margin)
    crop_ylim = (ymin - y_margin, ymax + y_margin)

    fig = plt.figure(figsize=(16, 8.4))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[80, 1],
        left=0.04, right=0.99, top=0.93, bottom=0.02,
        wspace=0.15, hspace=0.04,
        figure=fig,
    )
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])
    ax_prog = fig.add_subplot(gs[1, :])
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_xticks([])
    ax_prog.set_yticks([])
    for spine in ax_prog.spines.values():
        spine.set_visible(False)
    ax_prog.add_patch(Rectangle((0, 0), 1, 1, facecolor=PROGRESS_BG))
    prog_bar = ax_prog.add_patch(Rectangle((0, 0), 0, 1, facecolor=PROGRESS_FG))

    def update(frame_idx: int) -> list[Rectangle]:
        target = TARGETS[frame_idx // FRAMES_PER_TARGET]
        for eye in eyes:
            eye.look_at(target)
        ax1.clear()
        ax2.clear()
        plot_setup_and_camera_view(
            eyes, [target] * len(eyes), camera, [light],
            ax1=ax1, ax2=ax2, fig=fig, ref_bounds=REF_BOUNDS,
        )
        ax1.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        _apply_crop(ax2, crop_xlim, crop_ylim)
        prog_bar.set_width((frame_idx + 1) / TOTAL_FRAMES)
        return [prog_bar]

    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000 / FPS, blit=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_PATH, writer="ffmpeg", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
