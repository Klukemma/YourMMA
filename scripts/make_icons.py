"""Draw the app icons, with no image library.

A phone will only offer to install a site that ships icons, so the icons are
not decoration here - without them the install prompt never appears. Pillow is
not a dependency of this project and adding one to draw four squares would be
a poor trade, so this writes the PNGs directly: a PNG is a signature, an IHDR,
one zlib-compressed block of scanlines and an IEND, and nothing here needs
more than that.

    python3 scripts/make_icons.py

Writes app/icons/*.png. Re-run after changing the palette below; the files are
committed because a build step that needs Python is a build step the phone
cannot run.

THE MARK. A circle split on the diagonal, red corner against blue corner, with
a gap of background between them. It is the only thing in the app that has to
read at 48 pixels, so it is two shapes and one line, and it carries the
app's own --red and --blue rather than inventing a third palette.

Maskable icons get a smaller circle: Android crops a maskable icon to whatever
shape the launcher uses, and only the middle 80% is guaranteed to survive.
"""

import struct
import zlib
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
ICONS = APP / "icons"

# The app's own tokens, dark theme: --ground, --red, --blue.
BG = (0x14, 0x15, 0x1B)
RED = (0xE0, 0x4A, 0x4A)
BLUE = (0x4A, 0x8F, 0xE0)

SUPERSAMPLE = 4  # Edges at 48px are the whole job; 4x4 is enough to hide them.
GAP = 0.052      # Half-width of the background band, as a fraction of size.


def write_png(path, size, pixels):
    """Write RGB8 pixels (a flat bytearray, size*size*3) as a PNG."""
    raw = bytearray()
    stride = size * 3
    for y in range(size):
        raw.append(0)  # filter type 0 (None): these images compress fine flat
        raw.extend(pixels[y * stride:(y + 1) * stride])

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)
    return len(png)


def draw(size, radius):
    """The mark, supersampled. `radius` is the circle, as a fraction of size."""
    n = size * SUPERSAMPLE
    r = radius * n
    gap = GAP * n
    centre = (n - 1) / 2.0

    # Accumulate colour per output pixel, then divide. Working in ints would
    # band the edges; the sums are small enough that floats are exact anyway.
    acc = [[0.0, 0.0, 0.0] for _ in range(size * size)]

    for sy in range(n):
        dy = sy - centre
        row = (sy // SUPERSAMPLE) * size
        for sx in range(n):
            dx = sx - centre
            if dx * dx + dy * dy > r * r:
                colour = BG
            else:
                # Signed distance to the diagonal through the centre. The
                # diagonal runs top-left to bottom-right, so the red corner is
                # the lower-left half and the blue corner the upper-right.
                d = (dx - dy) * 0.7071067811865476
                colour = BG if abs(d) < gap else (BLUE if d > 0 else RED)
            cell = acc[row + (sx // SUPERSAMPLE)]
            cell[0] += colour[0]
            cell[1] += colour[1]
            cell[2] += colour[2]

    per = float(SUPERSAMPLE * SUPERSAMPLE)
    out = bytearray(size * size * 3)
    for i, cell in enumerate(acc):
        out[i * 3] = int(cell[0] / per + 0.5)
        out[i * 3 + 1] = int(cell[1] / per + 0.5)
        out[i * 3 + 2] = int(cell[2] / per + 0.5)
    return out


# (filename, pixels, circle radius). The maskable ones stay inside the 80%
# safe zone Android promises to keep; the others fill the tile.
PLAN = (
    ("icon-192.png", 192, 0.44),
    ("icon-512.png", 512, 0.44),
    ("maskable-192.png", 192, 0.34),
    ("maskable-512.png", 512, 0.34),
    ("apple-touch-icon.png", 180, 0.44),
)


def main():
    ICONS.mkdir(parents=True, exist_ok=True)
    for name, size, radius in PLAN:
        written = write_png(ICONS / name, size, draw(size, radius))
        print(f"wrote {ICONS / name} ({written:,} bytes)")


if __name__ == "__main__":
    main()
