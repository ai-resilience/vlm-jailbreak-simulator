"""Measured, lossless text layout for adapted dataset compositions.

This renderer deliberately does not change the paper reproduction renderers.
Text is wrapped by ink width, including unbroken identifiers, and is never
truncated. Layout failure raises rather than saving a clipped image.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import re
import unicodedata

from PIL import Image, ImageChops, ImageDraw, ImageFont


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


@lru_cache(maxsize=8)
def _coverage(path: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont
    font = TTFont(path, fontNumber=0)
    try:
        return frozenset(font.getBestCmap())
    finally:
        font.close()


@dataclass(frozen=True)
class Layout:
    text: str
    lines: list[str]
    line_boxes: list[list[int]]
    font_size: int
    font_path: str
    canvas: list[int]
    ink_box: list[int]
    margin: int
    text_sha256: str

    def to_dict(self):
        return asdict(self)


def _width(text, font):
    box = font.getbbox(text)
    return box[2] - box[0]


def wrap_pixels(text: str, font, width: int) -> list[str]:
    """Word wrap, with character wrapping for URLs/Base64/CJK, without slicing."""
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}" if current else word
            if _width(candidate, font) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            for char in word:
                if _width(char, font) > width:
                    raise ValueError("a single glyph is wider than the text area")
                if current and _width(current + char, font) > width:
                    lines.append(current)
                    current = ""
                current += char
        lines.append(current)
    if compact_text("\n".join(lines)) != compact_text(text):
        raise AssertionError("text was lost while wrapping")
    return lines


def render_text(
    text: str, *, font_path: str, width: int = 768, height: int = 768,
    font_size: int = 48, min_font_size: int = 28, margin: int = 32,
    max_height: int = 1536, fill: str = "black",
) -> tuple[Image.Image, Layout]:
    if not text.strip():
        raise ValueError("cannot render empty text")
    if not (width > 2 * margin > 0 and max_height >= height > 2 * margin):
        raise ValueError("invalid text canvas/margins")
    if not 1 <= min_font_size <= font_size:
        raise ValueError("invalid font size range")
    if any(unicodedata.category(c) == "Cc" and c not in "\n\t\r" for c in text):
        raise ValueError("unsupported control character in text")
    missing = sorted({ord(c) for c in text if not c.isspace()} - _coverage(font_path))
    if missing:
        raise ValueError(f"font lacks glyphs: {[hex(c) for c in missing]}")
    for size in range(font_size, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        lines = wrap_pixels(text, font, width - 2 * margin)
        ascent, descent = font.getmetrics()
        line_height = max(ascent + descent, size) + max(6, size // 5)
        required = 2 * margin + len(lines) * line_height
        if required <= height or size == min_font_size:
            break
    canvas_height = max(height, required)
    if canvas_height > max_height:
        raise ValueError(f"text needs {canvas_height}px height, limit is {max_height}; no output saved")
    image = Image.new("RGB", (width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    boxes = []
    for index, line in enumerate(lines):
        if not line:
            continue
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        x, y = margin - left, margin + index * line_height - top
        box = [x + left, y + top, x + right, y + bottom]
        if not (margin <= box[0] <= box[2] <= width - margin
                and margin <= box[1] <= box[3] <= canvas_height - margin):
            raise AssertionError(f"glyph bounds outside canvas: {box}")
        draw.text((x, y), line, font=font, fill=fill)
        boxes.append(box)
    ink = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    if ink is None or not (ink[0] >= margin and ink[1] >= margin
                           and ink[2] <= width - margin and ink[3] <= canvas_height - margin):
        raise AssertionError("raster ink extends beyond the protected text area")
    layout = Layout(text, lines, boxes, size, font_path, list(image.size), list(ink), margin,
                    hashlib.sha256(text.encode()).hexdigest())
    return image, layout


def validate_layout(layout: dict, image_size, *, inspection_max_side=2048, min_display_font=12):
    """Independent bounds/text-retention check after placement in a final image."""
    if compact_text("\n".join(layout["lines"])) != compact_text(layout["text"]):
        raise ValueError("layout text differs from its wrapped lines")
    if hashlib.sha256(layout["text"].encode()).hexdigest() != layout["text_sha256"]:
        raise ValueError("layout text digest mismatch")
    width, height = image_size
    for box in layout["line_boxes"]:
        if not (0 < box[0] < box[2] < width and 0 < box[1] < box[3] < height):
            raise ValueError(f"clipped final text bounds: {box} in {image_size}")
    effective = layout["font_size"] * min(1, inspection_max_side / max(image_size))
    if effective < min_display_font:
        raise ValueError(f"text too small at inspection size: {effective:.1f}px")
    return effective
