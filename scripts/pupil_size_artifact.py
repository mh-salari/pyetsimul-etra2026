"""Pupil-size artifact (PSA) animation for slide 12b: a 2x2 comparison.

Two separate eyes, each with its OWN calibration: one with this participant's linear
pupil decentration enabled, one with it disabled. Each is calibrated at the
constricted size, then the pupil dilates to the dilated size while the eye stays
fixed on the screen centre.

  Top row    (decentration ON):  the pupil decentres as it grows, so the calibrated
                                 tracker reports a growing gaze error -- the PSA.
  Bottom row (decentration OFF): the pupil only changes size, stays centred, and the
                                 reported gaze barely moves.

Columns: camera view (pupil outline + centre + glint) | gaze on screen, zoomed to the
error around the target. Both rows share x/y limits per column so they're comparable.
Sizes + coefficients come from img/pupil_decentration.json.

Output: ../img/pupil_size_artifact.mp4
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from pyetsimul.core import Eye
from pyetsimul.core.cornea import ConicCornea
from pyetsimul.core.pupil_decentration import PupilDecentrationConfig
from pyetsimul.gaze_mapping.stampe1993 import Stampe1993GazeModel
from pyetsimul.types import Position3D

from lab_setup import HV9_CALIBRATION_POINTS, build_camera, build_eyes, build_light

OUT_PATH = Path(__file__).parent.parent / "img" / "pupil_size_artifact.mp4"

TARGET = Position3D(0.0, 0.0, 0.0)  # screen centre; the eye stays fixed here
PUPIL_BOUNDARY_POINTS = 100  # smooth (perfect-circle) pupil boundary

# This participant's measured pupil sizes + linear decentration, from
# pupil_decentration_overlay.py. Run that script first to generate the JSON.
_DEC_JSON = Path(__file__).parent.parent / "img" / "pupil_decentration.json"
if not _DEC_JSON.exists():
    raise SystemExit(f"missing {_DEC_JSON}; run pupil_decentration_overlay.py first.")
_MEASUREMENT = json.loads(_DEC_JSON.read_text())["pairs"][0]["measurement"]
CALIB_DIAMETER = _MEASUREMENT["constricted_mm"]  # mm -- calibrate at the constricted size (~0 error)
MAX_DIAMETER = _MEASUREMENT["dilated_mm"]  # mm -- dilate to the dilated size
DECENTRATION_CX = _MEASUREMENT["glint"]["cx"]  # mm/mm -- this participant's linear decentration
DECENTRATION_CY = _MEASUREMENT["glint"]["cy"]

FPS = 30
HOLD_START = 12  # frames held at the calibration size (calibrated, ~0 error)
RAMP = 66  # frames dilating up (and the same count contracting back)
CAMERA_CROP_MARGIN = 0.15  # tight crop around the pupil/glint features
SCREEN_CROP_MARGIN = 0.20  # tight crop around the gaze error
SCREEN_CROP_FLOOR_MM = 5.0  # never zoom tighter than this around the target

# Deck palette (shared with the pupil slide and the pipeline videos).
PUPIL_COLOR = "#C0392B"  # pupil-centre marker
PUPIL_LINE_COLOR = "#000000"  # pupil outline -- black
REF_COLOR = "#999999"
GLINT_FACE = "#C9B891"
GLINT_EDGE = "#1A3D40"
TARGET_COLOR = "#1A3D40"
ESTIMATE_COLOR = "#C0392B"  # red -- marks the estimated (drifting) gaze location
PROGRESS_BG = "#eeeeee"
PROGRESS_FG = "#999999"


def _diameters() -> np.ndarray:
    """Pupil diameter per frame for a continuous ping-pong loop.

    Hold at the calibration size, dilate to the dilated size, then contract back.
    No end hold, so it loops smoothly; the up- and down-ramps drop their shared
    endpoints so neither the apex nor the loop seam repeats a frame.
    """
    return np.concatenate([
        np.full(HOLD_START, CALIB_DIAMETER),
        np.linspace(CALIB_DIAMETER, MAX_DIAMETER, RAMP),
        np.linspace(MAX_DIAMETER, CALIB_DIAMETER, RAMP + 1)[1:-1],
    ])


def _closed(points: list) -> tuple[list[float], list[float]]:
    """x, y lists for a Point2D boundary with the loop closed (as in the pupil slide)."""
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return xs + xs[:1], ys + ys[:1]


def _feature_bounds(images: list) -> tuple[float, float, float, float]:
    """Bounding box of all pupil/glint features across frames, for a stable crop."""
    xs: list[float] = []
    ys: list[float] = []
    for img in images:
        for p in [img.pupil_center, *(img.pupil_boundary or []), *(img.corneal_reflections or [])]:
            if p is not None:
                xs.append(p.x)
                ys.append(p.y)
    return min(xs), max(xs), min(ys), max(ys)


def _run_condition(camera, light, diameters: np.ndarray, eye_position: Position3D, decentration: bool) -> dict:
    """Build a fresh eye, calibrate it on its own, and roll the pupil ramp.

    Each condition is a fundamentally different eye with its own calibration, so the
    no-decentration row is NOT the decentration row with the shift removed.
    """
    eye = Eye(cornea=ConicCornea(), pupil_type="elliptical", pupil_boundary_points=PUPIL_BOUNDARY_POINTS)
    eye.position = eye_position

    # Calibrate at the constricted size with a centred pupil; enable decentration after.
    eye.set_pupil_diameter(CALIB_DIAMETER)
    et = Stampe1993GazeModel.create([camera], [light], HV9_CALIBRATION_POINTS)
    et.run_calibration(eye)
    if decentration:
        eye.decentration_config = PupilDecentrationConfig(
            enabled=True,
            model_name="wildenmann_2013",
            baseline_diameter=CALIB_DIAMETER,
            x_coeff=DECENTRATION_CX,
            y_coeff=DECENTRATION_CY,
        )

    dist = float(np.linalg.norm([eye_position.x - TARGET.x, eye_position.y - TARGET.y, eye_position.z - TARGET.z]))
    images: list = []
    pred_xs: list[float] = []
    pred_zs: list[float] = []
    edeg: list[float] = []
    for d in diameters:
        eye.set_pupil_diameter(float(d))
        prediction = et.estimate_gaze_at(eye, TARGET)
        images.append(camera.take_image(eye, [light], use_refraction=et.use_refraction, center_method="ellipse"))
        if prediction is None or prediction.gaze_point is None:
            px, pz = TARGET.x, TARGET.z
        else:
            px, pz = prediction.gaze_point.x, prediction.gaze_point.z
        pred_xs.append(px)
        pred_zs.append(pz)
        edeg.append(float(np.degrees(np.arctan2(np.hypot(px - TARGET.x, pz - TARGET.z), dist))))
    return {"images": images, "pred_xs": pred_xs, "pred_zs": pred_zs, "edeg": edeg, "calib_center": images[0].pupil_center}


def _draw_camera_view(ax: plt.Axes, image, calib_center) -> None:
    """Camera view: pupil outline + centre + glint, plus a reference at the calibration centre."""
    if image.pupil_boundary:
        bx, by = _closed(image.pupil_boundary)
        ax.plot(bx, by, "-", color=PUPIL_LINE_COLOR, linewidth=2, label="Pupil")
    if calib_center is not None:
        ax.plot(
            calib_center.x, calib_center.y, "+", color=REF_COLOR, markersize=11, markeredgewidth=1.6,
            label=f"Centre at {CALIB_DIAMETER:.2f} mm",
        )
    if image.pupil_center is not None:
        ax.plot(
            image.pupil_center.x, image.pupil_center.y, "x", color=PUPIL_COLOR, markersize=9, markeredgewidth=1.8,
            label="Pupil centre",
        )
    for glint in image.corneal_reflections or []:
        if glint is not None:
            ax.plot(
                glint.x, glint.y, "o", markerfacecolor=GLINT_FACE, markeredgecolor=GLINT_EDGE,
                markersize=9, markeredgewidth=1.0, label="Glint",
            )
    ax.set_aspect("equal")
    ax.grid(visible=True, alpha=0.3)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(loc="upper right", fontsize=8)


def _draw_screen_view(ax: plt.Axes, pred_x: float, pred_z: float) -> None:
    """Zoomed gaze panel: the fixed target and the estimated (drifting) gaze."""
    ax.scatter([TARGET.x], [TARGET.z], marker="+", s=150, c=TARGET_COLOR, linewidths=2.5, label="Target", zorder=6)
    ax.scatter([pred_x], [pred_z], marker="+", s=110, c=ESTIMATE_COLOR, linewidths=2.5, label="Estimated gaze", zorder=7)
    ax.set_aspect("equal")
    ax.grid(visible=True, alpha=0.3)
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Z position (mm)")
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    """Render the 2x2 PSA comparison: decentration on (top) vs off (bottom), own calibration each."""
    eyes = build_eyes()
    camera = build_camera(eyes)
    light = build_light()
    eye_position = eyes[0].position
    diameters = _diameters()

    on = _run_condition(camera, light, diameters, eye_position, decentration=True)
    off = _run_condition(camera, light, diameters, eye_position, decentration=False)

    # Shared camera crop across BOTH conditions; Y flipped (0 at bottom).
    xmin, xmax, ymin, ymax = _feature_bounds(on["images"] + off["images"])
    cam_cx, cam_cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    cam_half = max(xmax - xmin, ymax - ymin) / 2 * (1 + CAMERA_CROP_MARGIN)
    cam_xlim = (cam_cx - cam_half, cam_cx + cam_half)
    cam_ylim = (cam_cy - cam_half, cam_cy + cam_half)  # min..max -> 0 at bottom

    # Shared screen crop: zoom to the error around the target (same limits both rows).
    max_drift = max(
        np.hypot(px - TARGET.x, pz - TARGET.z)
        for cond in (on, off)
        for px, pz in zip(cond["pred_xs"], cond["pred_zs"], strict=False)
    )
    screen_half = max(max_drift * (1 + SCREEN_CROP_MARGIN), SCREEN_CROP_FLOOR_MM)
    err_xlim = (TARGET.x - screen_half, TARGET.x + screen_half)
    err_zlim = (TARGET.z - screen_half, TARGET.z + screen_half)

    fig = plt.figure(figsize=(15, 13), facecolor="white")
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[60, 60, 2], left=0.06, right=0.97, top=0.94, bottom=0.05, wspace=0.2, hspace=0.22, figure=fig
    )
    ax_cam_on = fig.add_subplot(gs[0, 0])
    ax_err_on = fig.add_subplot(gs[0, 1])
    ax_cam_off = fig.add_subplot(gs[1, 0])
    ax_err_off = fig.add_subplot(gs[1, 1])
    ax_prog = fig.add_subplot(gs[2, :])
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_xticks([])
    ax_prog.set_yticks([])
    for spine in ax_prog.spines.values():
        spine.set_visible(False)
    ax_prog.add_patch(Rectangle((0, 0), 1, 1, facecolor=PROGRESS_BG))
    prog = ax_prog.add_patch(Rectangle((0, 0), 0, 1, facecolor=PROGRESS_FG))

    rows = [
        (on, ax_cam_on, ax_err_on, "With decentration"),
        (off, ax_cam_off, ax_err_off, "Without decentration"),
    ]
    n = len(diameters)

    def update(i: int) -> None:
        for cond, ax_cam, ax_err, label in rows:
            ax_cam.clear()
            ax_err.clear()

            _draw_camera_view(ax_cam, cond["images"][i], cond["calib_center"])
            ax_cam.set_xlim(*cam_xlim)
            ax_cam.set_ylim(*cam_ylim)  # Y flipped
            ax_cam.set_title(f"{label} — camera view, pupil {diameters[i]:.1f} mm", fontsize=13, fontweight="bold")

            _draw_screen_view(ax_err, cond["pred_xs"][i], cond["pred_zs"][i])
            ax_err.set_xlim(*err_xlim)
            ax_err.set_ylim(*err_zlim)
            ax_err.set_title(f"{label} — gaze error {cond['edeg'][i]:.2f}°", fontsize=13, fontweight="bold")

        prog.set_width((i + 1) / n)

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / FPS, blit=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_PATH, writer="ffmpeg", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
