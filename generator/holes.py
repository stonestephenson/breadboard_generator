"""
Hole grid renderer — draws all terminal and power rail holes.

Calls grid.py for every hole position. Zero pixel math here.
"""

from PIL import Image, ImageDraw
from generator.grid import BreadboardGrid


def draw_holes(img: Image.Image, grid: BreadboardGrid) -> Image.Image:
    """
    Draw all holes (terminal + rail) onto the board image.

    Holes are slightly-square with rounded corners per the WB-102 spec.
    """
    draw = ImageDraw.Draw(img)
    spec = grid.spec
    hole_color = tuple(spec['holes']['color'])
    shadow_color = tuple(spec['holes']['shadow_color'])
    d = grid.hole_diameter_px
    corner_r = d * 0.25  # rounded corner radius

    all_holes = grid.all_terminal_holes() + grid.all_rail_holes()

    for row, col in all_holes:
        cx, cy = grid.hole_center(row, col)
        _draw_single_hole(draw, cx, cy, d, corner_r, hole_color, shadow_color)

    return img


def _draw_single_hole(
    draw: ImageDraw.ImageDraw,
    cx: float, cy: float,
    diameter: float,
    corner_radius: float,
    color: tuple[int, int, int],
    shadow_color: tuple[int, int, int],
) -> None:
    """Draw one hole as a small rounded square — small puncture in plastic."""
    half = diameter / 2

    # Single rounded square, no extra shadow border to keep holes small
    draw.rounded_rectangle(
        [cx - half, cy - half, cx + half, cy + half],
        radius=int(corner_radius),
        fill=color,
    )
