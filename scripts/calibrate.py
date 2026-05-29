"""Calibration-sweep video: calibrates one Stampe (1993) model per eye on the HV9
grid, then sweeps 4 off-grid quadrant targets via render_calibration_view.

Output: ../img/calibrate.mp4
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from pyetsimul.gaze_mapping.stampe1993 import Stampe1993GazeModel
from pyetsimul.types import Position3D, ScreenGeometry
from pyetsimul.visualization import compute_calibration_errors, render_calibration_view

from lab_setup import (
    CAL_HALF_H,
    CAL_HALF_W,
    HV9_CALIBRATION_POINTS,
    REF_BOUNDS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    VIEW_AZIM,
    VIEW_ELEV,
    build_camera,
    build_eyes,
    build_light,
)

OUT_PATH = Path(__file__).parent.parent / "img" / "calibrate.mp4"

# Video sweep: 4 quadrant midpoints (halfway between centre and each corner).
# Deliberately *not* on the HV9 calibration grid so the audience sees the eye
# track to points the model wasn't calibrated on.
VIDEO_TARGETS: list[Position3D] = [
    Position3D(-CAL_HALF_W / 2, 0.0, CAL_HALF_H / 2),
    Position3D(CAL_HALF_W / 2, 0.0, CAL_HALF_H / 2),
    Position3D(CAL_HALF_W / 2, 0.0, -CAL_HALF_H / 2),
    Position3D(-CAL_HALF_W / 2, 0.0, -CAL_HALF_H / 2),
]

FPS = 30
SECONDS_PER_TARGET = 1.5
FRAMES_PER_TARGET = int(SECONDS_PER_TARGET * FPS)
TOTAL_FRAMES = len(VIDEO_TARGETS) * FRAMES_PER_TARGET

PROGRESS_BG = "#eeeeee"
PROGRESS_FG = "#999999"

RIGHT_COLOR = "#2B5F66"
LEFT_COLOR = "#7BAE9B"


def main() -> None:
    """Render the binocular calibration sweep MP4."""
    eyes = build_eyes()
    right_eye, left_eye = eyes
    camera = build_camera(eyes)
    light = build_light()

    et_right = Stampe1993GazeModel.create([camera], [light], HV9_CALIBRATION_POINTS)
    et_right.run_calibration(right_eye)
    et_left = Stampe1993GazeModel.create([camera], [light], HV9_CALIBRATION_POINTS)
    et_left.run_calibration(left_eye)

    plane_info = et_right.plane_info
    predict_fns = [et_right.estimate_gaze_at, et_left.estimate_gaze_at]
    screen = ScreenGeometry(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, plane="xz")

    # Pre-compute static calibration data once (used per frame).
    cached_calib_data = [
        compute_calibration_errors(predict_fns[i], eyes[i], HV9_CALIBRATION_POINTS, plane_info)
        for i in range(len(eyes))
    ]

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[80, 1],
        left=0.05, right=0.97, top=0.92, bottom=0.04,
        wspace=0.18, hspace=0.04,
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
        target = VIDEO_TARGETS[frame_idx // FRAMES_PER_TARGET]

        render_calibration_view(
            ax1, ax2, target,
            eyes=eyes,
            predict_fns=predict_fns,
            calibration_points=HV9_CALIBRATION_POINTS,
            plane_info=plane_info,
            cameras=[camera],
            lights=[light],
            cached_calib_data=cached_calib_data,
            eye_labels=["Right", "Left"],
            eye_colors=[RIGHT_COLOR, LEFT_COLOR],
            screen=screen,
            ref_bounds_3d=REF_BOUNDS,
            xlim_2d=(-SCREEN_WIDTH / 2 - 30, SCREEN_WIDTH / 2 + 30),
            ylim_2d=(-SCREEN_HEIGHT / 2 - 30, SCREEN_HEIGHT / 2 + 30),
        )
        ax1.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

        prog_bar.set_width((frame_idx + 1) / TOTAL_FRAMES)
        return [prog_bar]

    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000 / FPS, blit=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_PATH, writer="ffmpeg", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
