"""
Component renderers — draws electronic components on the breadboard.

Each component class has a draw(img, grid) method.
All positioning via grid.py — zero pixel math here.
"""

import math
from PIL import Image, ImageDraw, ImageFont
from generator.grid import BreadboardGrid


def _get_font(size_px: float) -> ImageFont.FreeTypeFont:
    """Load a TrueType font for component silkscreen text, falling back to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=max(6, int(size_px)))
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


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


class Arduino:
    """Arduino/Metro Mini microcontroller board straddling the center gap.

    Has a dark PCB body, pin headers along both long sides that insert into
    terminal rows, and a USB connector protruding from one end. Pin row
    selection (default rows 'c' and 'h') gives the 0.7" pin spacing of the
    Adafruit Metro Mini.
    """

    def __init__(
        self,
        start_col: int,
        n_pins: int | None = None,
        top_row: str | None = None,
        bottom_row: str | None = None,
        usb_end: str = "left",
    ):
        """
        Args:
            start_col: Column of the leftmost pin pair (1-indexed).
            n_pins: Number of pin pairs along each long edge. Defaults to spec.
            top_row: Row letter for top-side pins (must be in top half a-e).
            bottom_row: Row letter for bottom-side pins (must be in bottom half f-j).
            usb_end: 'left' (USB at start_col side) or 'right' (at the far side).
        """
        if usb_end not in ("left", "right"):
            raise ValueError(f"usb_end must be 'left' or 'right', got {usb_end!r}")
        self.start_col = start_col
        self.n_pins = n_pins
        self.top_row = top_row
        self.bottom_row = bottom_row
        self.usb_end = usb_end

    def draw(self, img: Image.Image, grid: BreadboardGrid) -> Image.Image:
        """Draw the Arduino body, headers, pin pads, USB connector, silkscreen."""
        spec = grid.spec
        defaults = spec['component_defaults']['arduino']

        n_pins = self.n_pins if self.n_pins is not None else defaults['default_n_pins']
        top_row = self.top_row if self.top_row is not None else defaults['default_top_row']
        bottom_row = self.bottom_row if self.bottom_row is not None else defaults['default_bottom_row']
        end_col = self.start_col + n_pins - 1

        if top_row not in grid.rows_top:
            raise ValueError(
                f"top_row {top_row!r} must be in top half {grid.rows_top}"
            )
        if bottom_row not in grid.rows_bottom:
            raise ValueError(
                f"bottom_row {bottom_row!r} must be in bottom half {grid.rows_bottom}"
            )
        if end_col > grid.n_cols or self.start_col < 1:
            raise ValueError(
                f"Arduino spans cols {self.start_col}-{end_col}, "
                f"outside terminal range 1-{grid.n_cols}"
            )

        ppmm = grid.ppmm
        pad_long = defaults['body_padding_long_mm'] * ppmm
        pad_short = defaults['body_padding_short_mm'] * ppmm

        x_first, y_top = grid.hole_center(top_row, self.start_col)
        x_last, y_bottom = grid.hole_center(bottom_row, end_col)

        body_x1 = x_first - pad_long
        body_x2 = x_last + pad_long
        body_y1 = y_top - pad_short
        body_y2 = y_bottom + pad_short

        body_color = tuple(defaults['body_color'])
        header_color = tuple(defaults['header_strip_color'])
        pad_color = tuple(defaults['pin_pad_color'])
        usb_color = tuple(defaults['usb_color'])
        usb_outline = tuple(defaults['usb_outline_color'])
        silkscreen_color = tuple(defaults['silkscreen_color'])

        corner_r = max(1, int(round(defaults['body_corner_radius_mm'] * ppmm)))
        strip_w = defaults['header_strip_width_mm'] * ppmm
        pad_size = defaults['pin_pad_size_mm'] * ppmm
        usb_w_px = defaults['usb_width_mm'] * ppmm
        usb_l_px = defaults['usb_length_mm'] * ppmm
        usb_protrusion = defaults['usb_protrusion_mm'] * ppmm

        draw = ImageDraw.Draw(img)

        # PCB body — dark rounded rectangle
        draw.rounded_rectangle(
            [body_x1, body_y1, body_x2, body_y2],
            radius=corner_r,
            fill=body_color,
        )

        # Header strips along both long edges (where the pins live)
        strip_inset = pad_long * 0.3
        for y_strip in (y_top, y_bottom):
            draw.rectangle(
                [body_x1 + strip_inset, y_strip - strip_w / 2,
                 body_x2 - strip_inset, y_strip + strip_w / 2],
                fill=header_color,
            )

        # Pin pads — small light squares at each pin position
        half_pad = pad_size / 2
        for col in range(self.start_col, end_col + 1):
            for row in (top_row, bottom_row):
                cx, cy = grid.hole_center(row, col)
                draw.rectangle(
                    [cx - half_pad, cy - half_pad,
                     cx + half_pad, cy + half_pad],
                    fill=pad_color,
                )

        # USB connector — silver rectangle protruding from one end, centered vertically
        body_cy = (body_y1 + body_y2) / 2
        if self.usb_end == "left":
            usb_x1 = body_x1 - usb_protrusion
            usb_x2 = usb_x1 + usb_l_px
        else:
            usb_x2 = body_x2 + usb_protrusion
            usb_x1 = usb_x2 - usb_l_px

        usb_y1 = body_cy - usb_w_px / 2
        usb_y2 = body_cy + usb_w_px / 2

        usb_corner = max(1, int(round(0.5 * ppmm)))
        draw.rounded_rectangle(
            [usb_x1, usb_y1, usb_x2, usb_y2],
            radius=usb_corner,
            fill=usb_color,
            outline=usb_outline,
        )

        # Silkscreen text in the middle of the PCB
        text = defaults.get('silkscreen_text')
        if text:
            font = _get_font(strip_w * 0.7)
            tx = (body_x1 + body_x2) / 2
            ty = (body_y1 + body_y2) / 2
            draw.text((tx, ty), text, fill=silkscreen_color, font=font, anchor='mm')

        return img
