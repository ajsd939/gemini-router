"""Generate the Tauri bundle icons.

Tauri needs real PNG/ICO files at build time. Rather than commit binaries this
script draws them from scratch with nothing but the standard library.

    python tools/make_icons.py
"""

import os
import struct
import zlib

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src-tauri", "icons")

# Indigo -> cyan, matching the app's accent gradient.
TOP = (79, 70, 229)
BOTTOM = (6, 182, 212)


def _rounded_alpha(x, y, size, radius):
    """Coverage of a rounded square, sampled with 3x3 supersampling."""
    hits = 0
    samples = 0
    for sy in range(3):
        for sx in range(3):
            px = x + (sx + 0.5) / 3.0
            py = y + (sy + 0.5) / 3.0
            samples += 1
            # Distance outside the inset rectangle, clamped to the corners.
            dx = max(radius - px, px - (size - radius), 0.0)
            dy = max(radius - py, py - (size - radius), 0.0)
            if dx * dx + dy * dy <= radius * radius:
                hits += 1
    return hits / samples


def _distance_to_segment(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    length_sq = vx * vx + vy * vy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    cx, cy = ax + t * vx, ay + t * vy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def render(size):
    """Return raw RGBA bytes for a `size` x `size` icon."""
    radius = size * 0.22
    thickness = size * 0.088
    # A chevron ">" (a prompt caret) drawn in white.
    segments = [
        (0.34 * size, 0.29 * size, 0.63 * size, 0.50 * size),
        (0.63 * size, 0.50 * size, 0.34 * size, 0.71 * size),
    ]

    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            alpha_shape = _rounded_alpha(x, y, size, radius)
            if alpha_shape <= 0:
                row += b"\x00\x00\x00\x00"
                continue
            frac = (y + 0.5) / size
            r = int(TOP[0] + (BOTTOM[0] - TOP[0]) * frac)
            g = int(TOP[1] + (BOTTOM[1] - TOP[1]) * frac)
            b = int(TOP[2] + (BOTTOM[2] - TOP[2]) * frac)

            px, py = x + 0.5, y + 0.5
            glyph = min(_distance_to_segment(px, py, *seg) for seg in segments)
            if glyph <= thickness / 2:
                ink = 1.0
            elif glyph <= thickness / 2 + 1.0:
                ink = 1.0 - (glyph - thickness / 2)
            else:
                ink = 0.0
            if ink > 0:
                r = int(r + (255 - r) * ink)
                g = int(g + (255 - g) * ink)
                b = int(b + (255 - b) * ink)

            row += bytes((r, g, b, int(round(alpha_shape * 255))))
        rows.append(bytes(row))
    return b"".join(rows)


def to_png(size, rgba):
    def chunk(tag, payload):
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    raw = bytearray()
    stride = size * 4
    for y in range(size):
        raw.append(0)  # filter: none
        raw += rgba[y * stride : (y + 1) * stride]

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def to_ico(entries):
    """entries: list of (size, png_bytes)."""
    count = len(entries)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    directory = b""
    payload = b""
    for size, data in entries:
        dim = 0 if size >= 256 else size
        directory += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
        payload += data
    return header + directory + payload


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sizes = {
        "32x32.png": 32,
        "128x128.png": 128,
        "128x128@2x.png": 256,
        "icon.png": 512,
    }
    for name, size in sizes.items():
        path = os.path.join(OUT_DIR, name)
        with open(path, "wb") as fh:
            fh.write(to_png(size, render(size)))
        print("wrote", path)

    ico_entries = [(size, to_png(size, render(size))) for size in (16, 32, 48, 64, 256)]
    ico_path = os.path.join(OUT_DIR, "icon.ico")
    with open(ico_path, "wb") as fh:
        fh.write(to_ico(ico_entries))
    print("wrote", ico_path)


if __name__ == "__main__":
    main()
