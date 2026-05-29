"""PSA leading-slide figure: pupil decentration from two annotated eye crops.

Left panel: glint-aligned overlay of the constricted and dilated eye, so the
corneal reflection is fixed and only the pupil moves. Right panel: the two fitted
pupil ellipses and their centres at the same scale; the centre offset is the
decentration.

Each crop carries an eye-annotation-tool annotation (pupil ellipse + at least one
glint), read from a sibling <stem>_annotation.json or from a --project JSON.

Also prints the pupil diameters, P-CR shift, and linear decentration coefficients
(cx, cy in mm/mm = PyEtSimul x_coeff/y_coeff), and writes them to
pupil_decentration.json in the simulator's decentration schema.

Outputs: ../img/pupil_decentration_overlay.png, ../img/pupil_decentration.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.ndimage import shift as nd_shift

IMG_DIR = Path(__file__).parent.parent / "img"
DEFAULT_CONSTRICTED = IMG_DIR / "constricted.png"
DEFAULT_DILATED = IMG_DIR / "dilated.png"
OUT_PATH = IMG_DIR / "pupil_decentration_overlay.png"

PX_PER_MM = 20.832  # cam2 right-eye px/mm calibration
CONSTRICTED_COLOR = "#1f77b4"  # blue
DILATED_COLOR = "#d62728"  # red
EYE_SLOT_PREFERENCE = ("single", "right", "left")
CROP_LIMBUS_RADII = 1.1  # crop half-width as a multiple of the limbus radius (>1 keeps the limbus in frame)


def _read_gray(image_path: Path) -> np.ndarray:
    """Load a PNG as a float grayscale array in [0, 1]."""
    img = mpimg.imread(str(image_path))
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.2989, 0.5870, 0.1140])
    return img.astype(np.float64)


def _load_detections(image_path: Path, project_path: Path | None) -> dict:
    """Return the ``detections`` dict for one image (sibling JSON, else project JSON)."""
    sidecar = image_path.with_name(f"{image_path.stem}_annotation.json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
    elif project_path is not None:
        proj = json.loads(project_path.read_text())
        data = (proj.get("images") or {}).get(str(image_path))
        if not data:
            raise SystemExit(f"{image_path.name} not found (or empty) in project {project_path}.")
    else:
        raise SystemExit(
            f"no annotation for {image_path.name}: expected sibling {sidecar.name} or pass --project.",
        )
    detections = data.get("detections") or {}
    if not detections:
        raise SystemExit(f"{image_path.name} has no 'detections' block; annotate it in eye-annotation-tool first.")
    return detections


def _find_block(detections: dict, needle: str) -> dict:
    """Return the first detection block whose plugin key contains ``needle``."""
    for key, block in detections.items():
        if needle in key.lower():
            return block
    raise SystemExit(f"no '{needle}' detection block found; available plugins: {sorted(detections)}.")


def _pick_eye(block: dict, eye: str | None) -> tuple[str, dict]:
    """Return ``(slot, result)`` for one detection block.

    Monocular annotations store ``result`` directly in the block (alongside
    ``id``/``params``); binocular ones nest it under a per-eye slot. Handle both.
    """
    if isinstance(block.get("result"), dict) and block["result"]:
        return "single", block["result"]
    slots = (eye,) if eye else EYE_SLOT_PREFERENCE
    for slot in slots:
        entry = block.get(slot)
        if isinstance(entry, dict) and entry.get("result"):
            return slot, entry["result"]
    raise SystemExit(f"no usable result in block (tried {list(slots)}); keys: {sorted(block)}.")


def _read_limbus(detections: dict, slot: str) -> tuple[tuple[float, float], float] | None:
    """Return ``(center, radius)`` for the limbus, or None if not annotated."""
    block = next((b for k, b in detections.items() if "limbus" in k.lower()), None)
    if not isinstance(block, dict):
        return None
    result = block.get("result") if isinstance(block.get("result"), dict) else (block.get(slot) or {}).get("result")
    if not result or "center" not in result:
        return None
    r_theta = result.get("R_theta")
    radius = float(np.mean(r_theta)) if r_theta else float(result.get("radius", 0.0))
    if radius <= 0:
        return None
    return (float(result["center"][0]), float(result["center"][1])), radius


def _parse_ellipse(ellipse: object) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return ``(center, size, angle)`` from either the dict or [center, size, angle] list form."""
    if isinstance(ellipse, dict):
        center, size, angle = ellipse["center"], ellipse["size"], ellipse["angle"]
    else:  # OpenCV RotatedRect serialised as [[cx, cy], [w, h], angle]
        center, size, angle = ellipse[0], ellipse[1], ellipse[2]
    return (float(center[0]), float(center[1])), (float(size[0]), float(size[1])), float(angle)


def read_eye(image_path: Path, eye: str | None, project_path: Path | None) -> dict:
    """Read the pupil ellipse and mean glint for one annotated eye crop."""
    detections = _load_detections(image_path, project_path)
    slot, pupil_result = _pick_eye(_find_block(detections, "pupil"), eye)
    _, glint_result = _pick_eye(_find_block(detections, "glint"), slot)

    center, size, angle = _parse_ellipse(pupil_result["ellipse"])
    glints = [(float(g["center"][0]), float(g["center"][1])) for g in glint_result.get("glints", [])]
    if not glints:
        raise SystemExit(f"{image_path.name} ({slot}) has no glints; at least one is needed for alignment.")

    mean_glint = (sum(g[0] for g in glints) / len(glints), sum(g[1] for g in glints) / len(glints))
    limbus = _read_limbus(detections, slot)
    return {
        "slot": slot,
        "ellipse_center": center,
        "ellipse_size": size,
        "ellipse_angle": angle,
        "mean_glint": mean_glint,
        "limbus_center": limbus[0] if limbus else None,
        "limbus_radius": limbus[1] if limbus else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Build the two-panel pupil-decentration figure from the two annotated crops."""
    parser = argparse.ArgumentParser(prog="pupil_decentration_overlay", description=__doc__.splitlines()[0])
    parser.add_argument("--constricted", type=Path, default=DEFAULT_CONSTRICTED, help="Constricted-pupil eye crop.")
    parser.add_argument("--dilated", type=Path, default=DEFAULT_DILATED, help="Dilated-pupil eye crop.")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--eye", choices=("single", "right", "left"), help="Force the eye slot (default: auto).")
    parser.add_argument("--project", type=Path, help="eye-annotation-tool project JSON (if no sibling files).")
    args = parser.parse_args(argv)

    con_img = _read_gray(args.constricted)
    dil_img = _read_gray(args.dilated)
    if con_img.shape != dil_img.shape:
        raise SystemExit(f"crops differ in size: constricted {con_img.shape} vs dilated {dil_img.shape}.")
    con = read_eye(args.constricted, args.eye, args.project)
    dil = read_eye(args.dilated, args.eye, args.project)

    # Glint-align constricted onto dilated: translate so the mean glints coincide
    # (CR-only alignment, mirroring how the EyeLink fixes the corneal reflection).
    dx = dil["mean_glint"][0] - con["mean_glint"][0]
    dy = dil["mean_glint"][1] - con["mean_glint"][1]
    con_aligned = nd_shift(con_img, shift=(dy, dx), order=1, mode="constant", cval=0.0)
    con_center = (con["ellipse_center"][0] + dx, con["ellipse_center"][1] + dy)
    dil_center = dil["ellipse_center"]

    fig, (ax_overlay, ax_ellipse) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Both panels share one window around the limbus, so the eye image and the
    #     ellipse plot are at the same scale (a pupil reads the same size in each) ---
    overlay = 0.5 * dil_img + 0.5 * con_aligned
    h, w = overlay.shape
    if dil["limbus_radius"]:
        cx, cy = dil["limbus_center"]
        half = CROP_LIMBUS_RADII * dil["limbus_radius"]  # crop radius (limbus radius x factor)
        x0, x1 = max(0, round(cx - half)), min(w, round(cx + half))
        y0, y1 = max(0, round(cy - half)), min(h, round(cy + half))
    else:
        x0, x1, y0, y1 = 0, w, 0, h

    # extent maps array pixels to image coords (origin upper-left, y down)
    ax_overlay.imshow(overlay[y0:y1, x0:x1], cmap="gray", vmin=0.0, vmax=1.0, extent=(x0, x1, y1, y0))
    ax_overlay.set_title("Eyes overlay (glint-aligned)", fontsize=12)
    ax_overlay.axis("off")

    # --- Right: the two pupil ellipses + centres in the glint-aligned frame ---
    ax_ellipse.add_patch(
        Ellipse(
            dil_center,
            *dil["ellipse_size"],
            angle=dil["ellipse_angle"],
            fill=False,
            edgecolor=DILATED_COLOR,
            linewidth=2,
        ),
    )
    ax_ellipse.add_patch(
        Ellipse(
            con_center,
            *con["ellipse_size"],
            angle=con["ellipse_angle"],
            fill=False,
            edgecolor=CONSTRICTED_COLOR,
            linewidth=2,
        ),
    )
    ax_ellipse.plot(*dil_center, marker=".", markersize=9, color=DILATED_COLOR, linestyle="none")
    ax_ellipse.plot(*con_center, marker=".", markersize=9, color=CONSTRICTED_COLOR, linestyle="none")

    shift_px = float(np.hypot(con_center[0] - dil_center[0], con_center[1] - dil_center[1]))
    shift_mm = shift_px / PX_PER_MM
    ax_ellipse.set_title(f"Pupil-centre shift = {shift_mm:.2f} mm ({shift_px:.1f} px)", fontsize=12)

    # Same window + equal aspect as the left panel -> identical scale; no frame box.
    ax_ellipse.set_xlim(x0, x1)
    ax_ellipse.set_ylim(y1, y0)  # inverted: image coords
    ax_ellipse.set_aspect("equal")
    ax_ellipse.set_xticks([])
    ax_ellipse.set_yticks([])
    for spine in ax_ellipse.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=CONSTRICTED_COLOR,
            markeredgewidth=2,
            linestyle="none",
            label="Constricted pupil",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=DILATED_COLOR,
            markeredgewidth=2,
            linestyle="none",
            label="Dilated pupil",
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            markersize=12,
            color=CONSTRICTED_COLOR,
            linestyle="none",
            label="Constricted pupil center",
        ),
        Line2D(
            [0], [0], marker=".", markersize=12, color=DILATED_COLOR, linestyle="none", label="Dilated pupil center"
        ),
    ]
    ax_ellipse.legend(handles=legend_handles, loc="upper right", fontsize=10)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}  (pupil-centre shift {shift_px:.1f} px; slots con={con['slot']}, dil={dil['slot']})")

    # Linear decentration: signed P-CR change (dilated - glint-aligned constricted)
    # over the pupil-diameter change.
    con_diam_px = (con["ellipse_size"][0] + con["ellipse_size"][1]) / 2
    dil_diam_px = (dil["ellipse_size"][0] + dil["ellipse_size"][1]) / 2
    delta_diam_px = dil_diam_px - con_diam_px
    dx_px = dil_center[0] - con_center[0]
    dy_px = dil_center[1] - con_center[1]
    cx, cy = dx_px / delta_diam_px, dy_px / delta_diam_px
    con_mm, dil_mm = con_diam_px / PX_PER_MM, dil_diam_px / PX_PER_MM
    delta_mm = dil_mm - con_mm
    dx_mm, dy_mm = dx_px / PX_PER_MM, dy_px / PX_PER_MM

    decentration = {
        "participant": "salari_real_eye",
        "trial": "1",
        "camera": "cam2",
        "px_per_mm": {"right_eye": PX_PER_MM},
        "pairs": [
            {
                "source": "target_frames",
                "hv9_position": 0,
                "eye": "right_eye",
                "measurement": {
                    "dilated_diameter_px": dil_diam_px,
                    "constricted_diameter_px": con_diam_px,
                    "delta_diameter_px": delta_diam_px,
                    "dilated_mm": dil_mm,
                    "constricted_mm": con_mm,
                    "delta_mm": delta_mm,
                    "glint": {"cx": cx, "cy": cy, "dx_px": dx_px, "dy_px": dy_px, "dx_mm": dx_mm, "dy_mm": dy_mm},
                },
            },
        ],
    }
    json_path = args.out.with_name("pupil_decentration.json")
    json_path.write_text(json.dumps(decentration, indent=2))

    print(f"diam: constricted={con_mm:.2f} mm, dilated={dil_mm:.2f} mm")
    print(f"pupil-diameter change      = {delta_mm:.2f} mm  ({delta_diam_px:.1f} px)")
    print(f"pupil-centre shift (P-CR)  = ({dx_mm:.3f}, {dy_mm:.3f}) mm  ({dx_px:.1f}, {dy_px:.1f} px)")
    print(f"linear decentration coeffs = cx={cx:.3f}, cy={cy:.3f}  mm/mm  (= PyEtSimul x_coeff/y_coeff)")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
