import gzip
import json
from pathlib import Path

from PIL import Image, PngImagePlugin


def write_comment_png(path: Path, payload: dict) -> Path:
    img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "NovelAI")
    info.add_text("Comment", json.dumps(payload))
    img.save(path, "PNG", pnginfo=info)
    return path


def write_stealth_png(path: Path, payload: dict, *, compressed: bool = True) -> Path:
    raw = json.dumps(payload).encode("utf-8")
    if compressed:
        body = gzip.compress(raw)
        magic = b"stealth_pngcomp"
    else:
        body = raw
        magic = b"stealth_pnginfo"
    packed = magic + (len(body) * 8).to_bytes(4, "big") + body
    bits = []
    for byte in packed:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    # 64x64 alpha LSB holds 512 bytes; small NAI JSON + gzip fits.
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    pixels = img.load()
    width, height = img.size
    index = 0
    done = False
    for x in range(width):
        for y in range(height):
            if index >= len(bits):
                done = True
                break
            r, g, b, a = pixels[x, y]
            pixels[x, y] = (r, g, b, (a & ~1) | bits[index])
            index += 1
        if done:
            break
    img.save(path, "PNG")
    return path


def write_webp_usercomment(path: Path, payload: dict) -> Path:
    img = Image.new("RGB", (8, 8), (255, 0, 0))
    exif = Image.Exif()
    exif[0x9286] = json.dumps(payload)
    img.save(path, "WEBP", exif=exif)
    return path
