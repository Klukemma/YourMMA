"""Draw the YourNFL app icons, with no image library.

The MMA app's icon writer (scripts/make_icons.py) with a different mark: a
football, point to point on the diagonal, in the app's red with white laces,
on the dark ground. Like the MMA mark it has to read at 48 pixels, so it is
one shape and a few lines.

    python3 scripts/make_nfl_icons.py

Writes nfl/app/icons/*.png (committed: the phone cannot run Python).
Maskable icons get a smaller ball so it survives Android's 80% safe zone.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_icons import BG, RED, SUPERSAMPLE, write_png  # noqa: E402

ICONS = Path(__file__).resolve().parent.parent / "nfl" / "app" / "icons"
LACE = (0xF4, 0xF1, 0xEA)


def draw(size, scale):
    """The ball, supersampled. `scale` is its half-length as a fraction of size."""
    n = size * SUPERSAMPLE
    centre = (n - 1) / 2.0
    a = scale * n                 # half-length, tip to tip along the diagonal
    w = a * 0.55                  # half-width at the middle
    # The ball is the lens where two circles overlap: radius R, centres c
    # either side of the long axis, chosen so the lens is 2a long and 2w wide.
    c = (a * a - w * w) / (2 * w)
    R = c + w
    lace_len, lace_w = a * 0.42, n * 0.016
    stitch_half, stitch_w = a * 0.13, n * 0.014
    stitches = [-0.27, -0.09, 0.09, 0.27]
    acc = [[0.0, 0.0, 0.0] for _ in range(size * size)]
    for sy in range(n):
        dy = sy - centre
        row = (sy // SUPERSAMPLE) * size
        for sx in range(n):
            dx = sx - centre
            u = (dx - dy) * 0.70710678      # along the ball, bottom-left to top-right
            v = (dx + dy) * 0.70710678
            inside = u * u + (abs(v) + c) ** 2 <= R * R
            colour = BG
            if inside:
                colour = RED
                if abs(v) < lace_w and abs(u) < lace_len:
                    colour = LACE
                for s in stitches:
                    if abs(u - s * a) < stitch_w and abs(v) < stitch_half:
                        colour = LACE
            px = acc[row + sx // SUPERSAMPLE]
            px[0] += colour[0]; px[1] += colour[1]; px[2] += colour[2]
    k = SUPERSAMPLE * SUPERSAMPLE
    out = bytearray()
    for r, g, b in acc:
        out += bytes((int(round(r / k)), int(round(g / k)), int(round(b / k))))
    return out


def main():
    ICONS.mkdir(parents=True, exist_ok=True)
    for name, size, scale in (("icon-192.png", 192, 0.40), ("icon-512.png", 512, 0.40),
                              ("maskable-192.png", 192, 0.30), ("maskable-512.png", 512, 0.30),
                              ("apple-touch-icon.png", 180, 0.40)):
        nbytes = write_png(ICONS / name, size, draw(size, scale))
        print(f"   {name}: {size}x{size}, {nbytes:,} bytes")


if __name__ == "__main__":
    main()
