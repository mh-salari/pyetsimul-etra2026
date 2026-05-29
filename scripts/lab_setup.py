"""Shared lab geometry for the slide scripts (EyeLink 1000 Plus setup).

PyEtSimul right-handed frame, origin at the screen centre, in mm: x = horizontal
(+ = participant's right), y = distance from the screen plane (+ = toward the
participant), z = vertical (+ = up).
"""

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.cornea import ConicCornea
from pyetsimul.types import Position3D

# Screen (1920 x 1080 display).
SCREEN_WIDTH = 531.36
SCREEN_HEIGHT = 298.98
SCREEN_CENTRE_ABOVE_TABLE = 345.0

# Eyes (binocular, ±30 mm from screen centre, 835 mm perpendicular).
EYE_X_RIGHT = 30.0
EYE_X_LEFT = -30.0
EYE_Y = 835.0
EYE_Z = 430.0 - SCREEN_CENTRE_ABOVE_TABLE  # 85.0

# EyeLink 1000 Plus IR camera (below the screen).
CAMERA_X = -100.0
CAMERA_Y = 420.0
CAMERA_Z = 220.0 - SCREEN_CENTRE_ABOVE_TABLE  # -125.0

# EyeLink illuminator (mounted on the camera bar).
LIGHT_X = 95.0
LIGHT_Y = 435.0
LIGHT_Z = 230.0 - SCREEN_CENTRE_ABOVE_TABLE  # -115.0

# HV9 calibration area: 88% x 83% of the screen.
CAL_HALF_W = 0.88 * SCREEN_WIDTH / 2
CAL_HALF_H = 0.83 * SCREEN_HEIGHT / 2

# HV9 calibration grid: centre + 4 cardinal edges + 4 corners.
HV9_CALIBRATION_POINTS: list[Position3D] = [
    Position3D(0.0, 0.0, 0.0),
    Position3D(0.0, 0.0, CAL_HALF_H),
    Position3D(0.0, 0.0, -CAL_HALF_H),
    Position3D(-CAL_HALF_W, 0.0, 0.0),
    Position3D(CAL_HALF_W, 0.0, 0.0),
    Position3D(-CAL_HALF_W, 0.0, CAL_HALF_H),
    Position3D(CAL_HALF_W, 0.0, CAL_HALF_H),
    Position3D(-CAL_HALF_W, 0.0, -CAL_HALF_H),
    Position3D(CAL_HALF_W, 0.0, -CAL_HALF_H),
]

# Shared 3D view (so all slide images render with the same axes / camera angle).
REF_BOUNDS: dict[str, tuple[float, float]] = {
    "x": (-300, 300),
    "y": (0, 900),
    "z": (-200, 200),
}
VIEW_ELEV = 20
VIEW_AZIM = -60


def build_eyes() -> list[Eye]:
    """Binocular pair of conic-cornea eyes at the lab positions. No look_at applied."""
    right_eye = Eye(cornea=ConicCornea())
    right_eye.position = Position3D(EYE_X_RIGHT, EYE_Y, EYE_Z)
    left_eye = Eye(cornea=ConicCornea())
    left_eye.position = Position3D(EYE_X_LEFT, EYE_Y, EYE_Z)
    return [right_eye, left_eye]


def build_camera(eyes: list[Eye]) -> Camera:
    """EyeLink 1000 Plus camera, positioned and pointed at the binocular midpoint.

    Expects ``eyes`` in the order returned by ``build_eyes()`` ([right, left]).
    """
    camera = Camera()
    camera.position = Position3D(CAMERA_X, CAMERA_Y, CAMERA_Z)
    # point_at_binocular(left, right)
    camera.point_at_binocular(eyes[1].position, eyes[0].position)
    return camera


def build_light() -> Light:
    """EyeLink illuminator at the lab position."""
    return Light(position=Position3D(LIGHT_X, LIGHT_Y, LIGHT_Z))
