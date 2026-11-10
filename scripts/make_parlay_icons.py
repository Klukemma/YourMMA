"""Draw the YourMulti (same-day parlays) app icons, with no image library.

The MMA app's icon writer (scripts/make_icons.py) with a different mark: a
ticket of three legs - three rounded bars, each with its dot, the last one
in the family's red - on the dark ground. Like the other marks it has to
read at 48 pixels, so it is a few plain shapes.

    python3 scripts/make_parlay_icons.py

Writes parlays/app/icons/*.png (committed: the phone cannot run Python).
Maskable icons get a smaller mark so it survives Android's 80% safe zone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_icons import BG, RED, SUPERSAMPLE, write_png  # noqa: E402

ICONS = Path(__file__).resolve().parent.parent / "parlays" / "app" / "icons"
INK = (0xF4, 0xF1, 0xEA)


def _bar(x, y, x0, x1, yc, h):
    """Inside a horizontal capsule from x0 to x1 centred on yc, height h."""
    r = h / 2.0
    if x0 + r <= x <= x1 - r:
        return abs(y - yc) <= r
    cx = x0 + r if x < x0 + r else x1 - r
    return (x - cx) ** 2 + (y - yc) ** 2 <= r * r


def draw(size, scale):
    n = size * SUPERSAMPLE
    w, h = scale * n * 2.0, scale * n * 0.36       # bar width and height
    gap = h * 0.62
    x0 = (n - w) / 2.0
    ys = [n / 2.0 - (h + gap), n / 2.0, n / 2.0 + (h + gap)]
    dot_r = h * 0.30
    acc = [[0.0, 0.0, 0.0] for _ in range(size * size)]
    for sy in range(n):
        for sx in range(n):
            colour = BG
            for i, yc in enumerate(ys):
                if _bar(sx, sy, x0, x0 + w, yc, h):
                    colour = RED if i == 2 else INK
                    dx = sx - (x0 + h / 2.0)
                    if dx * dx + (sy - yc) ** 2 <= dot_r * dot_r:
                        colour = BG
                    break
            a = acc[(sy // SUPERSAMPLE) * size + sx // SUPERSAMPLE]
            a[0] += colour[0]
            a[1] += colour[1]
            a[2] += colour[2]
    k = SUPERSAMPLE * SUPERSAMPLE
    out = bytearray()
    for r, g, b in acc:
        out += bytes((int(round(r / k)), int(round(g / k)), int(round(b / k))))
    return out


def main():
    ICONS.mkdir(parents=True, exist_ok=True)
    for name, size, scale in (("icon-192.png", 192, 0.33), ("icon-512.png", 512, 0.33),
                              ("maskable-192.png", 192, 0.25), ("maskable-512.png", 512, 0.25),
                              ("apple-touch-icon.png", 180, 0.33)):
        nbytes = write_png(ICONS / name, size, draw(size, scale))
        print(f"   {name}: {size}x{size}, {nbytes:,} bytes")


if __name__ == "__main__":
    main()
