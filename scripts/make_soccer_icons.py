"""Draw the YourSoccer app icons, with no image library.

The MMA app's icon writer (scripts/make_icons.py) with a third mark: a
ball in the apps' red, its panels cut in the dark ground - a pentagon in
the middle, five more clipped by the rim, seams joining them. Like the
other marks it has to read at 48 pixels, so it is one shape and a few lines.

    python3 scripts/make_soccer_icons.py

Writes soccer/app/icons/*.png (committed: the phone cannot run Python).
Maskable icons get a smaller ball so it survives Android's 80% safe zone.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_icons import BG, RED, SUPERSAMPLE, write_png  # noqa: E402

ICONS = Path(__file__).resolve().parent.parent / "soccer" / "app" / "icons"


def pentagon(cx, cy, radius, rotation):
    return [(cx + radius * math.sin(rotation + k * 2 * math.pi / 5),
             cy - radius * math.cos(rotation + k * 2 * math.pi / 5)) for k in range(5)]


def inside(poly, x, y):
    """Point in a convex polygon whose vertices run clockwise on screen."""
    sign = None
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        s = cross >= 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def near_segment(x, y, a, b, width):
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
    px, py = x1 + t * dx, y1 + t * dy
    return (x - px) ** 2 + (y - py) ** 2 <= width * width


def draw(size, scale):
    """The ball, supersampled. `scale` is its radius as a fraction of size."""
    n = size * SUPERSAMPLE
    c = (n - 1) / 2.0
    r = scale * n
    middle = pentagon(c, c, r * 0.33, 0.0)
    rim = [pentagon(c + r * 1.02 * math.sin(k * 2 * math.pi / 5), c - r * 1.02 * math.cos(k * 2 * math.pi / 5),
                    r * 0.30, math.pi + k * 2 * math.pi / 5) for k in range(5)]
    # A seam runs from each corner of the middle panel out to its rim
    # panel, and from each rim panel round to the next, along the hexagons.
    out_at = [(c + r * 0.74 * math.sin(k * 2 * math.pi / 5), c - r * 0.74 * math.cos(k * 2 * math.pi / 5))
              for k in range(5)]
    seams = [(middle[k], out_at[k]) for k in range(5)]
    seams += [(out_at[k], (c + r * 0.80 * math.sin((k + 0.5) * 2 * math.pi / 5),
                           c - r * 0.80 * math.cos((k + 0.5) * 2 * math.pi / 5))) for k in range(5)]
    seams += [(out_at[(k + 1) % 5], (c + r * 0.80 * math.sin((k + 0.5) * 2 * math.pi / 5),
                                     c - r * 0.80 * math.cos((k + 0.5) * 2 * math.pi / 5))) for k in range(5)]
    seams += [((c + r * 0.80 * math.sin((k + 0.5) * 2 * math.pi / 5), c - r * 0.80 * math.cos((k + 0.5) * 2 * math.pi / 5)),
               (c + r * 1.0 * math.sin((k + 0.5) * 2 * math.pi / 5), c - r * 1.0 * math.cos((k + 0.5) * 2 * math.pi / 5)))
              for k in range(5)]
    seam_w = n * 0.012
    acc = [[0.0, 0.0, 0.0] for _ in range(size * size)]
    for sy in range(n):
        row = (sy // SUPERSAMPLE) * size
        for sx in range(n):
            colour = BG
            if (sx - c) ** 2 + (sy - c) ** 2 <= r * r:
                colour = RED
                if inside(middle, sx, sy) or any(inside(p, sx, sy) for p in rim) \
                        or any(near_segment(sx, sy, a, b, seam_w) for a, b in seams):
                    colour = BG
            px = acc[row + sx // SUPERSAMPLE]
            px[0] += colour[0]; px[1] += colour[1]; px[2] += colour[2]
    k = SUPERSAMPLE * SUPERSAMPLE
    out = bytearray()
    for red, green, blue in acc:
        out += bytes((int(round(red / k)), int(round(green / k)), int(round(blue / k))))
    return out


def main():
    ICONS.mkdir(parents=True, exist_ok=True)
    for name, size, scale in (("icon-192.png", 192, 0.40), ("icon-512.png", 512, 0.40),
                              ("maskable-192.png", 192, 0.31), ("maskable-512.png", 512, 0.31),
                              ("apple-touch-icon.png", 180, 0.40)):
        nbytes = write_png(ICONS / name, size, draw(size, scale))
        print(f"   {name}: {size}x{size}, {nbytes:,} bytes")


if __name__ == "__main__":
    main()
