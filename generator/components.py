"""
Component renderers — draws electronic components on the breadboard.

Each component class has a draw(img, grid) method.
All positioning via grid.py — zero pixel math here.
"""

import math
from PIL import Image, ImageDraw
from generator.grid import BreadboardGrid


class Resistor:
    """Axial resistor with colored body, band stripes, and two wire legs."""

    def __init__(
        self,
        leg1: tuple[str, int],
        leg2: tuple[str, int],
        bands: list[tuple[int, int, int]] | None = None,
    ):
        """
        Args:
            leg1: (row, col) for first leg insertion point.
            leg2: (row, col) for second leg insertion point.
            bands: List of RGB color tuples for resistance bands.
                   Defaults to brown-black-brown-gold (100 ohm).
        """
        self.leg1 = leg1
        self.leg2 = leg2
        self.bands = bands or [
            (139, 69, 19),   # brown
            (30, 30, 30),    # black
            (139, 69, 19),   # brown
            (212, 175, 55),  # gold
        ]

    def draw(self, img: Image.Image, grid: BreadboardGrid) -> Image.Image:
        """Draw the resistor onto the board image."""
        draw = ImageDraw.Draw(img)
        spec = grid.spec
        defaults = spec['component_defaults']

        body_len_px = defaults['resistor_body_length_mm'] * grid.ppmm
        body_w_px = defaults['resistor_body_width_mm'] * grid.ppmm
        lead_d_px = defaults['resistor_lead_diameter_mm'] * grid.ppmm

        x1, y1 = grid.hole_center(self.leg1[0], self.leg1[1])
        x2, y2 = grid.hole_center(self.leg2[0], self.leg2[1])

        # Angle and midpoint between the two legs
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Unit vectors along and perpendicular to the resistor axis
        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular

        # Draw leads (thin lines from leg holes to body edges)
        body_half = body_len_px / 2
        body_start_x = mx - ux * body_half
        body_start_y = my - uy * body_half
        body_end_x = mx + ux * body_half
        body_end_y = my + uy * body_half

        lead_w = max(1, int(round(lead_d_px)))
        lead_color = (180, 180, 180)  # silver leads
        draw.line([(x1, y1), (body_start_x, body_start_y)],
                  fill=lead_color, width=lead_w)
        draw.line([(body_end_x, body_end_y), (x2, y2)],
                  fill=lead_color, width=lead_w)

        # Draw body as a rotated rectangle
        hw = body_w_px / 2
        corners = [
            (body_start_x + px * hw, body_start_y + py * hw),
            (body_end_x + px * hw, body_end_y + py * hw),
            (body_end_x - px * hw, body_end_y - py * hw),
            (body_start_x - px * hw, body_start_y - py * hw),
        ]
        body_color = (210, 180, 140)  # tan/beige resistor body
        draw.polygon(corners, fill=body_color, outline=(160, 140, 100))

        # Draw color bands evenly spaced along the body
        n_bands = len(self.bands)
        if n_bands > 0:
            band_width = body_len_px * 0.08
            spacing = body_len_px / (n_bands + 1)
            for i, band_color in enumerate(self.bands):
                t = spacing * (i + 1) - body_half
                bx = mx + ux * t
                by = my + uy * t
                half_bw = band_width / 2
                band_corners = [
                    (bx - ux * half_bw + px * hw, by - uy * half_bw + py * hw),
                    (bx + ux * half_bw + px * hw, by + uy * half_bw + py * hw),
                    (bx + ux * half_bw - px * hw, by + uy * half_bw - py * hw),
                    (bx - ux * half_bw - px * hw, by - uy * half_bw - py * hw),
                ]
                draw.polygon(band_corners, fill=band_color)

        return img


class LED:
    """Light-emitting diode with dome body and two wire legs."""

    def __init__(
        self,
        anode: tuple[str, int],
        cathode: tuple[str, int],
        color: str = "red",
    ):
        """
        Args:
            anode: (row, col) for anode (longer leg, +).
            cathode: (row, col) for cathode (shorter leg, -).
            color: LED color name (from spec led_colors).
        """
        self.anode = anode
        self.cathode = cathode
        self.color = color

    def draw(self, img: Image.Image, grid: BreadboardGrid) -> Image.Image:
        """Draw the LED onto the board image."""
        draw = ImageDraw.Draw(img)
        spec = grid.spec
        defaults = spec['component_defaults']

        dome_d_px = defaults['led_dome_diameter_mm'] * grid.ppmm
        lead_d_px = defaults['resistor_lead_diameter_mm'] * grid.ppmm

        # Resolve LED color
        led_colors = defaults['led_colors']
        rgb = tuple(led_colors[self.color])

        ax, ay = grid.hole_center(self.anode[0], self.anode[1])
        cx, cy = grid.hole_center(self.cathode[0], self.cathode[1])

        # Dome center is midpoint between anode and cathode
        mx, my = (ax + cx) / 2, (ay + cy) / 2

        # Draw legs
        lead_w = max(1, int(round(lead_d_px)))
        lead_color = (180, 180, 180)
        draw.line([(ax, ay), (mx, my)], fill=lead_color, width=lead_w)
        draw.line([(cx, cy), (mx, my)], fill=lead_color, width=lead_w)

        # Draw dome (filled circle) with semi-transparent tint
        r = dome_d_px / 2
        # Outer dome
        draw.ellipse(
            [mx - r, my - r, mx + r, my + r],
            fill=rgb,
            outline=(max(0, rgb[0] - 40), max(0, rgb[1] - 40), max(0, rgb[2] - 40)),
        )

        # Inner highlight for 3D effect
        highlight_r = r * 0.4
        highlight_color = (
            min(255, rgb[0] + 80),
            min(255, rgb[1] + 80),
            min(255, rgb[2] + 80),
        )
        draw.ellipse(
            [mx - highlight_r - r * 0.15, my - highlight_r - r * 0.15,
             mx + highlight_r - r * 0.15, my + highlight_r - r * 0.15],
            fill=highlight_color,
        )

        return img
