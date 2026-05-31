"""Render code-snippet PNGs for the deck, in the img/code/ atom-one-dark style.

Pygments tokenises the Python; PIL draws a rounded dark box (#101c1d) with line
numbers and atom-one-dark syntax colours on a transparent margin. A snippet can
be rendered whole, or as build-up steps that highlight one group of lines and dim
the rest (like the setup imports -> eyes -> camera -> light sequence).

Outputs: ../img/code/<name>.png
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String

from lab_setup import SWEEP_TARGETS

IMG_DIR = Path(__file__).parent.parent / "img" / "code"

# atom-one-dark palette (box bg is the deck's dark-teal override, not #282c34)
BG = (16, 28, 29)  # #101c1d
DEFAULT_FG = "#abb2bf"
LINE_NUMBER = "#5c6370"
DIM = "#454b57"  # colour for lines outside the current build-up step (darker than line numbers)
SOFT_DIM = "#7a8394"  # gentle dim, to push a changing line forward without hiding the rest
COLORS = {
    Comment: "#5c6370",
    Keyword: "#c678dd",
    Name.Builtin: "#56b6c2",
    Name.Builtin.Pseudo: "#e5c07b",
    Name.Function: "#61afef",
    Name.Class: "#e5c07b",
    Name.Decorator: "#61afef",
    String: "#98c379",
    Number: "#d19a66",
    Operator: "#abb2bf",
    Punctuation: "#abb2bf",
    Name: "#abb2bf",
}

FONT_SIZE = 40
LINE_SPACING = 1.4
PAD = 44  # inside the box
MARGIN = 30  # transparent border around the box
RADIUS = 22
GUTTER_GAP = 28  # between line numbers and code

SETUP_CODE = """from pyetsimul.core import Camera, Eye, Light
from pyetsimul.types import Position3D

right_eye = Eye()
right_eye.position = Position3D(30, 835, 85)
left_eye = Eye()
left_eye.position = Position3D(-30, 835, 85)

camera = Camera()
camera.position = Position3D(-100, 420, -125)
camera.point_at_binocular(left_eye.position,
                          right_eye.position)

light = Light(position=Position3D(95, 435, -115))"""

# Build-up: each step highlights its 1-based lines; the rest are dimmed.
SETUP_STEPS = {
    "setup_imports": {1, 2},
    "setup_eyes": {4, 5, 6, 7},
    "setup_camera": {9, 10, 11, 12},
    "setup_light": {14},
}

def _pupil_glint_code(x: int, y: int, z: int) -> str:
    """The pupil+glint for-loop snippet for one gaze target (only line 1 changes)."""
    return f"""target = Position3D({x}, {y}, {z})
for eye in (left_eye, right_eye):
    eye.look_at(target)
    image = camera.take_image(eye, [light])
    image.pupil_center          # pupil centre   (px)
    image.corneal_reflections   # glint (CR)     (px)
    image.pupil_boundary        # pupil outline  (px)"""


def _font(size: int) -> ImageFont.FreeTypeFont:
    """A bold monospace face, preferring the macOS Menlo the originals used."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size, index=1)  # Menlo Bold
    except OSError:
        pass
    import matplotlib.font_manager as fm

    return ImageFont.truetype(fm.findfont(fm.FontProperties(family="monospace", weight="bold")), size)


def _color(ttype) -> str:
    """Resolve a token type to a hex colour, walking up the token hierarchy."""
    t = ttype
    while t is not None:
        if t in COLORS:
            return COLORS[t]
        t = t.parent
    return DEFAULT_FG


def _tokenise(code: str) -> list[list[tuple[str, str]]]:
    """Return per-line lists of (text, colour) spans."""
    lines: list[list[tuple[str, str]]] = [[]]
    for ttype, value in lex(code, PythonLexer()):
        color = _color(ttype)
        parts = value.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append([])
            if part:
                lines[-1].append((part, color))
    while len(lines) > 1 and not lines[-1]:  # drop the trailing newline's empty line
        lines.pop()
    return lines


def render(
    code: str,
    out_path: Path,
    highlight: set[int] | None = None,
    soft_dim: set[int] | None = None,
    start_line: int = 1,
) -> None:
    """Render one snippet.

    ``highlight`` keeps colour only on those 1-based lines (others fully dimmed);
    ``soft_dim`` gently greys those 1-based lines; ``start_line`` sets the first
    line number so a snippet can read as a continuation.
    """
    font = _font(FONT_SIZE)
    char_w = font.getlength("0")
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * LINE_SPACING)

    spans = _tokenise(code)
    n = len(spans)
    num_w = len(str(start_line + n - 1)) * char_w
    code_x = MARGIN + PAD + num_w + GUTTER_GAP

    cols = max((sum(len(t) for t, _ in line) for line in spans), default=0)
    box_w = int(code_x - MARGIN + cols * char_w + PAD)
    box_h = int(2 * PAD + n * line_h)

    img = Image.new("RGBA", (box_w + 2 * MARGIN, box_h + 2 * MARGIN), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([MARGIN, MARGIN, MARGIN + box_w, MARGIN + box_h], radius=RADIUS, fill=BG)

    for i, line in enumerate(spans):
        y = MARGIN + PAD + i * line_h
        num = str(start_line + i)
        draw.text((MARGIN + PAD + num_w - font.getlength(num), y), num, font=font, fill=LINE_NUMBER)
        if highlight is not None and (i + 1) not in highlight:
            override = DIM
        elif soft_dim is not None and (i + 1) in soft_dim:
            override = SOFT_DIM
        else:
            override = None
        x = code_x
        for text, color in line:
            draw.text((x, y), text, font=font, fill=override or color)
            x += font.getlength(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Wrote {out_path}")


def main() -> None:
    """Render the setup build-up, the static pupil+glint snippet, and the per-gaze series."""
    for name, lines in SETUP_STEPS.items():
        render(SETUP_CODE, IMG_DIR / f"{name}.png", highlight=lines)
    render(_pupil_glint_code(0, 0, 0), IMG_DIR / "pupil_glint.png", start_line=16)
    for i, target in enumerate(SWEEP_TARGETS):
        code = _pupil_glint_code(round(target.x), round(target.y), round(target.z))
        render(code, IMG_DIR / f"demo_gaze_{i}.png", soft_dim={2, 3, 4, 5, 6, 7}, start_line=16)


if __name__ == "__main__":
    main()
