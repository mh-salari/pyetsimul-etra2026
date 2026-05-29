"""Pipeline videos for slides 11 and 12: calibrate a Stampe (1993) model on the
right eye, then run a screen test (gaze sweeps the screen) and an observer test
(eye position sweeps a +/-50 mm grid).

Each is a side-by-side MP4: the 3D setup (left, moving target/eye) and the error
quiver (right, one arrow added per measurement, read from the same gaze_result
that is .pprint()-ed). Output: ../img/screen_test.mp4, ../img/observer_test.mp4
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.evaluation.gaze_accuracy import GazeAccuracyResult, evaluate_gaze_accuracy
from pyetsimul.gaze_mapping.stampe1993 import Stampe1993GazeModel
from pyetsimul.geometry.plane_detection import PlaneInfo
from pyetsimul.simulation import DataGenerationStrategy, EyePositionVariation, TargetPositionVariation
from pyetsimul.types import Point3D, Position3D, ScreenGeometry
from pyetsimul.types.geometry import Point2D
from pyetsimul.visualization.coordinate_utils import prepare_eye_data_for_plots
from pyetsimul.visualization.gaze_accuracy_plots import detect_variation_plane, extract_variation_coords
from pyetsimul.visualization.setup_plots import plot_setup

from lab_setup import (
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

IMG_DIR = Path(__file__).parent.parent / "img"
SCREEN_OUT = IMG_DIR / "screen_test.mp4"
OBSERVER_OUT = IMG_DIR / "observer_test.mp4"

GRID_SIDE = 10
OBSERVER_RANGE_MM = 50.0
FPS = 15

# Persian Gulf palette (shared with the rest of the deck).
ARROW_COLOR = "#2B5F66"  # older arrows / settled error field
LATEST_COLOR = "#C9B891"  # sand: the arrow just added (matches the moving marker)
BASE_POINT_COLOR = "#5E8A85"
PROGRESS_BG = "#eeeeee"
PROGRESS_FG = "#999999"
ARROW_MAX_RATIO = 0.3  # longest arrow as a fraction of the plot range (matches plot_error_vectors_2d)
SMALL_ERROR_MAGNIFY = 10  # modest fixed magnification for errors too small to see (screen test)


def _collect_points(result: GazeAccuracyResult, plane_info: PlaneInfo, *, is_target: bool) -> list[dict]:
    """Per-point animation data in dataset order, valid predictions only.

    Each dict: base (arrow position), uv (predicted-actual gaze, mm), target, eye, deg.
    """
    if is_target:
        primary, secondary = plane_info.primary_axis, plane_info.secondary_axis
    else:
        primary, secondary, _, _ = detect_variation_plane(result.variation)

    points: list[dict] = []
    for i in range(len(result.ground_truth_points)):
        predicted = result.predicted_points[i]
        if predicted is None:
            continue
        actual = result.ground_truth_points[i]
        actual_2d = plane_info.extract_2d_coords(actual)
        predicted_2d = plane_info.extract_2d_coords(predicted)
        eye_pos = result.observer_positions[i]
        base = actual_2d if is_target else extract_variation_coords(eye_pos, primary, secondary)
        points.append(
            {
                "base": (base[0], base[1]),
                "uv": (predicted_2d[0] - actual_2d[0], predicted_2d[1] - actual_2d[1]),
                "target": actual,
                "eye": eye_pos,
                "deg": result.errors_angular[i],
            }
        )
    return points


def _render_video(
    out_path: Path,
    points: list[dict],
    *,
    is_target: bool,
    eye: Eye,
    camera: Camera,
    light: Light,
    screen: ScreenGeometry,
    plane_info: PlaneInfo,
    left_title: str,
    right_title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Animate left setup + right accumulating error quiver, and save as MP4."""
    bx = np.array([p["base"][0] for p in points])
    by = np.array([p["base"][1] for p in points])
    u = np.array([p["uv"][0] for p in points])
    v = np.array([p["uv"][1] for p in points])
    deg = np.array([p["deg"] for p in points])

    # One scale for the whole field so arrows don't rescale per frame: shrink any
    # that would overflow the plot, and magnify errors too small to see (screen
    # test) by a fixed modest factor rather than exaggerating them to fit.
    magnitudes = np.sqrt(u**2 + v**2)
    max_mag = magnitudes.max()
    plot_range = max(bx.max() - bx.min(), by.max() - by.min())
    target_len = plot_range * ARROW_MAX_RATIO
    scale = target_len / max_mag if max_mag > 0 else 1.0
    if scale > 1.0:
        scale = SMALL_ERROR_MAGNIFY
        scale_label = f"   (arrows ×{scale:.0f})"
    else:
        scale_label = ""
    us, vs = u * scale, v * scale

    # Fixed right-panel limits with margin for arrow tips.
    mx = (bx.max() - bx.min()) * 0.18 + abs(us).max()
    my = (by.max() - by.min()) * 0.18 + abs(vs).max()
    xlim_2d = (bx.min() - mx, bx.max() + mx)
    ylim_2d = (by.min() - my, by.max() + my)

    calib_2d = [Point2D(*plane_info.extract_2d_coords(p)) for p in HV9_CALIBRATION_POINTS]

    fig = plt.figure(figsize=(20, 9), facecolor="white")
    gs = gridspec.GridSpec(
        1, 2,
        left=0.04, right=0.97, top=0.90, bottom=0.13, wspace=0.16, figure=fig,
    )
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_2d = fig.add_subplot(gs[0, 1])
    # Progress bar placed manually at the very bottom, clear of the axis labels.
    ax_prog = fig.add_axes((0.04, 0.045, 0.93, 0.02))
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_xticks([])
    ax_prog.set_yticks([])
    for spine in ax_prog.spines.values():
        spine.set_visible(False)
    ax_prog.add_patch(Rectangle((0, 0), 1, 1, facecolor=PROGRESS_BG))
    prog_bar = ax_prog.add_patch(Rectangle((0, 0), 0, 1, facecolor=PROGRESS_FG))

    n = len(points)

    def update(frame: int) -> None:
        point = points[frame]

        # ---- Left: 3D setup with the moving element ----
        ax_3d.clear()
        if is_target:
            target_3d = Point3D(point["target"].x, 0.0, point["target"].z)
        else:
            eye.position = point["eye"]
            target_3d = Point3D(0.0, 0.0, 0.0)
        prepared = prepare_eye_data_for_plots([eye], [target_3d], [light], [camera])
        plot_setup(
            ax_3d,
            prepared["eyes_data"],
            [target_3d],
            [light],
            [camera],
            prepared["cr_3d_lists"],
            calib_points=calib_2d,
            screen=screen,
            ref_bounds=REF_BOUNDS,
        )
        ax_3d.scatter([target_3d.x], [0], [target_3d.z], c=LATEST_COLOR, s=60, marker="+", zorder=10)
        ax_3d.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        ax_3d.set_title(left_title, fontsize=14, fontweight="bold", pad=10)

        # ---- Right: error quiver, accumulating one arrow per point ----
        ax_2d.clear()
        upto = frame + 1
        ax_2d.scatter(bx[:upto], by[:upto], s=10, c=BASE_POINT_COLOR, alpha=0.6, zorder=2)
        if frame > 0:
            ax_2d.quiver(
                bx[:frame], by[:frame], us[:frame], vs[:frame],
                scale=1, scale_units="xy", angles="xy", width=0.004, color=ARROW_COLOR, zorder=3,
            )
        ax_2d.quiver(
            [bx[frame]], [by[frame]], [us[frame]], [vs[frame]],
            scale=1, scale_units="xy", angles="xy", width=0.006, color=LATEST_COLOR, zorder=4,
        )
        ax_2d.set_xlim(*xlim_2d)
        ax_2d.set_ylim(*ylim_2d)
        ax_2d.set_aspect("equal")
        ax_2d.grid(visible=True, alpha=0.3)
        ax_2d.set_xlabel(xlabel)
        ax_2d.set_ylabel(ylabel)
        running_mean = float(np.mean(deg[:upto]))
        ax_2d.set_title(
            f"{right_title}\nPoints: {upto}/{n}   Mean error: {running_mean:.2f}°{scale_label}",
            fontsize=13, fontweight="bold", pad=10,
        )

        prog_bar.set_width(upto / n)

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / FPS, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer="ffmpeg", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    """Calibrate Stampe (1993) on the right eye, run both tests, save both videos."""
    eyes = build_eyes()
    right_eye = eyes[0]
    original_eye_pos = right_eye.position
    camera = build_camera(eyes)
    light = build_light()
    screen = ScreenGeometry(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, plane="xz")

    # --- Calibrate: one Stampe (1993) model on the right eye ---
    et = Stampe1993GazeModel.create([camera], [light], HV9_CALIBRATION_POINTS)
    et.run_calibration(right_eye)
    plane_info = et.plane_info

    calib = accuracy_at_calibration_points(et, eye=right_eye)
    calib.pprint("Stampe 1993 — HV9 Calibration (right eye)")

    # --- Screen test: gaze over the full screen ---
    screen_variation = TargetPositionVariation(
        grid_center=Position3D(0.0, 0.0, 0.0),
        dx=[-SCREEN_WIDTH / 2, SCREEN_WIDTH / 2],
        dy=[0.0, 0.0],
        dz=[-SCREEN_HEIGHT / 2, SCREEN_HEIGHT / 2],
        grid_size=[GRID_SIDE, 1, GRID_SIDE],
    )
    data_gen = DataGenerationStrategy(
        eyes=[right_eye],
        cameras=[camera],
        lights=[light],
        gaze_target=Position3D(0.0, 0.0, 0.0),
        experiment_name="screen_test",
        save_to_file=False,
        use_refraction=et.use_refraction,
    )
    screen_dataset = data_gen.execute(screen_variation)
    screen_results = evaluate_gaze_accuracy(eye_tracker=et, dataset=screen_dataset, description="Screen test")
    screen_results.pprint("Stampe 1993 — Screen Test (right eye)")

    # --- Observer test: eye position movement, gaze fixed on screen centre ---
    observer_variation = EyePositionVariation(
        center=original_eye_pos,
        dx=[-OBSERVER_RANGE_MM, OBSERVER_RANGE_MM],
        dy=[-OBSERVER_RANGE_MM, OBSERVER_RANGE_MM],
        dz=[0.0, 0.0],
        grid_size=[GRID_SIDE, GRID_SIDE, 1],
    )
    data_gen.set_experiment_name("observer_test")
    observer_dataset = data_gen.execute(observer_variation)
    observer_results = evaluate_gaze_accuracy(eye_tracker=et, dataset=observer_dataset, description="Observer test")
    observer_results.pprint("Stampe 1993 — Observer Test (right eye)")

    # Both datasets are generated; animations below mutate the eye for rendering
    # only, so they must run after all eye.serialize() calls inside execute().
    right_eye.position = original_eye_pos  # eye is fixed for the screen test
    _render_video(
        SCREEN_OUT,
        _collect_points(screen_results, plane_info, is_target=True),
        is_target=True,
        eye=right_eye,
        camera=camera,
        light=light,
        screen=screen,
        plane_info=plane_info,
        left_title="Screen test — gaze sweeps the screen",
        right_title="Gaze error across the screen",
        xlabel=f"Target {plane_info.primary_axis.upper()} position (mm)",
        ylabel=f"Target {plane_info.secondary_axis.upper()} position (mm)",
    )

    primary, secondary, _, _ = detect_variation_plane(observer_results.variation)
    _render_video(
        OBSERVER_OUT,
        _collect_points(observer_results, plane_info, is_target=False),
        is_target=False,
        eye=right_eye,
        camera=camera,
        light=light,
        screen=screen,
        plane_info=plane_info,
        left_title="Observer test — head position moves",
        right_title="Gaze error vs. eye position",
        xlabel=f"Observer {primary.upper()} position (mm)",
        ylabel=f"Observer {secondary.upper()} position (mm)",
    )


if __name__ == "__main__":
    main()
