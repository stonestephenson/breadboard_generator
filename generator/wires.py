"""
Wire routing and drawing — draws colored wires between breadboard holes.

Supports straight and L-shaped routing. All positioning via grid.py.
"""

from PIL import Image, ImageDraw
from generator.grid import BreadboardGrid


def draw_wire(
    img: Image.Image,
    grid: BreadboardGrid,
    start: tuple[str, int],
    end: tuple[str, int],
    color: str | tuple[int, int, int] = "red",
    routing: str = "straight",
) -> Image.Image:
    """
    Draw a wire between two holes on the breadboard.

    Args:
        img: Board image to draw on.
        grid: BreadboardGrid for coordinate lookup.
        start: (row, col) of wire start hole.
        end: (row, col) of wire end hole.
        color: Color name (from spec) or RGB tuple.
        routing: 'straight' or 'L' for L-shaped routing.

    Returns:
        Modified image.
    """
    draw = ImageDraw.Draw(img)
    spec = grid.spec

    # Resolve color
    if isinstance(color, str):
        wire_colors = spec['component_defaults']['wire_colors']
        rgb = tuple(wire_colors[color])
    else:
        rgb = color

    thickness = spec['component_defaults']['wire_thickness_mm'] * grid.ppmm

    sx, sy = grid.hole_center(start[0], start[1])
    ex, ey = grid.hole_center(end[0], end[1])

    if routing == "straight":
        _draw_straight(draw, sx, sy, ex, ey, thickness, rgb)
    elif routing == "L":
        _draw_l_shaped(draw, sx, sy, ex, ey, thickness, rgb)
    else:
        raise ValueError(f"Unknown routing type: {routing!r}")

    return img


def _draw_straight(
    draw: ImageDraw.ImageDraw,
    sx: float, sy: float, ex: float, ey: float,
    thickness: float, color: tuple[int, int, int],
) -> None:
    """Draw a straight wire between two points."""
    draw.line([(sx, sy), (ex, ey)], fill=color, width=max(1, int(round(thickness))))


def _draw_l_shaped(
    draw: ImageDraw.ImageDraw,
    sx: float, sy: float, ex: float, ey: float,
    thickness: float, color: tuple[int, int, int],
) -> None:
    """Draw an L-shaped wire: horizontal first, then vertical."""
    w = max(1, int(round(thickness)))
    # Go horizontal to the end column, then vertical to the end row
    draw.line([(sx, sy), (ex, sy)], fill=color, width=w)
    draw.line([(ex, sy), (ex, ey)], fill=color, width=w)
