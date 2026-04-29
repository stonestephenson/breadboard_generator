"""Unit tests for generator/components.py and generator/wires.py."""

import os
import numpy as np
from PIL import Image
from generator.grid import BreadboardGrid, load_spec
from generator.board import draw_board_base
from generator.holes import draw_holes
from generator.wires import draw_wire
from generator.components import Resistor, LED, Arduino

SPEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'board_spec.json')


def _make_board():
    """Create a supersampled board with grid for testing."""
    spec = load_spec(SPEC_PATH)
    ss = spec['rendering']['supersample_factor']
    grid = BreadboardGrid(spec, ppmm_override=spec['board']['pixels_per_mm'] * ss)
    img = draw_board_base(grid)
    img = draw_holes(img, grid)
    return img, grid, spec


class TestWires:
    def test_straight_wire_draws_pixels(self):
        img, grid, _ = _make_board()
        arr_before = np.array(img).copy()
        draw_wire(img, grid, ('a', 10), ('a', 20), color='red', routing='straight')
        arr_after = np.array(img)
        assert not np.array_equal(arr_before, arr_after), "Wire should change pixels"

    def test_l_shaped_wire_draws_pixels(self):
        img, grid, _ = _make_board()
        arr_before = np.array(img).copy()
        draw_wire(img, grid, ('a', 10), ('e', 15), color='blue', routing='L')
        arr_after = np.array(img)
        assert not np.array_equal(arr_before, arr_after), "L-wire should change pixels"

    def test_wire_color_from_spec(self):
        img, grid, spec = _make_board()
        draw_wire(img, grid, ('a', 5), ('a', 10), color='green', routing='straight')
        arr = np.array(img)
        # Sample midpoint between cols 5 and 10 on row a
        cx1, cy = grid.hole_center('a', 7)
        px, py = int(round(cx1)), int(round(cy))
        pixel = arr[py, px]
        # Green channel should be dominant
        assert pixel[1] > pixel[0] and pixel[1] > pixel[2], \
            f"Expected green-dominant pixel, got {tuple(pixel[:3])}"

    def test_wire_rgb_tuple_color(self):
        img, grid, _ = _make_board()
        draw_wire(img, grid, ('f', 5), ('f', 10), color=(200, 40, 40), routing='straight')
        arr = np.array(img)
        cx, cy = grid.hole_center('f', 7)
        px, py = int(round(cx)), int(round(cy))
        pixel = arr[py, px]
        assert pixel[0] > pixel[1] and pixel[0] > pixel[2], \
            f"Expected red-dominant pixel, got {tuple(pixel[:3])}"

    def test_invalid_routing_raises(self):
        img, grid, _ = _make_board()
        try:
            draw_wire(img, grid, ('a', 1), ('a', 5), color='red', routing='zigzag')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestResistor:
    def test_resistor_draws_pixels(self):
        img, grid, _ = _make_board()
        arr_before = np.array(img).copy()
        r = Resistor(leg1=('a', 10), leg2=('a', 15))
        r.draw(img, grid)
        arr_after = np.array(img)
        assert not np.array_equal(arr_before, arr_after), "Resistor should change pixels"

    def test_resistor_body_at_midpoint(self):
        """Resistor body should have tan/beige pixels near the midpoint."""
        img, grid, _ = _make_board()
        r = Resistor(leg1=('a', 20), leg2=('a', 25))
        r.draw(img, grid)
        arr = np.array(img)
        # Midpoint between cols 20 and 25
        cx1, _ = grid.hole_center('a', 22)
        _, cy = grid.hole_center('a', 22)
        px, py = int(round(cx1)), int(round(cy))
        pixel = arr[py, px]
        # Body color is tan (210, 180, 140) — R > G > B
        assert pixel[0] > pixel[2], \
            f"Expected warm-toned body pixel, got {tuple(pixel[:3])}"

    def test_resistor_vertical(self):
        """Resistor should render vertically across the center gap."""
        img, grid, _ = _make_board()
        r = Resistor(leg1=('c', 40), leg2=('h', 40))
        r.draw(img, grid)
        arr = np.array(img)
        # Check that pixels changed at the center gap area
        cx, _ = grid.hole_center('c', 40)
        _, cy_e = grid.hole_center('e', 40)
        _, cy_f = grid.hole_center('f', 40)
        mid_y = int(round((cy_e + cy_f) / 2))
        pixel = arr[mid_y, int(round(cx))]
        # Should not be white (board body) — resistor body or lead should be there
        assert not all(c > 240 for c in pixel[:3]), \
            f"Expected non-white pixel at center gap, got {tuple(pixel[:3])}"


class TestLED:
    def test_led_draws_pixels(self):
        img, grid, _ = _make_board()
        arr_before = np.array(img).copy()
        led = LED(anode=('a', 45), cathode=('b', 45), color='red')
        led.draw(img, grid)
        arr_after = np.array(img)
        assert not np.array_equal(arr_before, arr_after), "LED should change pixels"

    def test_led_color_at_dome(self):
        """LED dome center should have the specified color."""
        img, grid, spec = _make_board()
        led = LED(anode=('a', 45), cathode=('b', 45), color='red')
        led.draw(img, grid)
        arr = np.array(img)
        # Dome is at midpoint between anode and cathode
        ax, ay = grid.hole_center('a', 45)
        cx, cy = grid.hole_center('b', 45)
        mx, my = int(round((ax + cx) / 2)), int(round((ay + cy) / 2))
        pixel = arr[my, mx]
        # Red LED should have dominant red channel
        assert pixel[0] > 150 and pixel[0] > pixel[1] and pixel[0] > pixel[2], \
            f"Expected red-dominant dome pixel, got {tuple(pixel[:3])}"

    def test_led_green(self):
        img, grid, _ = _make_board()
        led = LED(anode=('f', 50), cathode=('g', 50), color='green')
        led.draw(img, grid)
        arr = np.array(img)
        ax, ay = grid.hole_center('f', 50)
        cx, cy = grid.hole_center('g', 50)
        mx, my = int(round((ax + cx) / 2)), int(round((ay + cy) / 2))
        pixel = arr[my, mx]
        assert pixel[1] > pixel[0] and pixel[1] > pixel[2], \
            f"Expected green-dominant dome pixel, got {tuple(pixel[:3])}"

    def test_led_blue(self):
        """Blue LED should render with blue-dominant dome pixels."""
        img, grid, _ = _make_board()
        led = LED(anode=('a', 30), cathode=('b', 30), color='blue')
        led.draw(img, grid)
        arr = np.array(img)
        ax, ay = grid.hole_center('a', 30)
        cx, cy = grid.hole_center('b', 30)
        mx, my = int(round((ax + cx) / 2)), int(round((ay + cy) / 2))
        pixel = arr[my, mx]
        assert pixel[2] > pixel[0] and pixel[2] > pixel[1], \
            f"Expected blue-dominant dome pixel, got {tuple(pixel[:3])}"

    def test_led_white(self):
        """White LED should render with very bright, near-balanced dome pixels."""
        img, grid, _ = _make_board()
        led = LED(anode=('h', 35), cathode=('i', 35), color='white')
        led.draw(img, grid)
        arr = np.array(img)
        ax, ay = grid.hole_center('h', 35)
        cx, cy = grid.hole_center('i', 35)
        mx, my = int(round((ax + cx) / 2)), int(round((ay + cy) / 2))
        pixel = arr[my, mx]
        # White dome: every channel should be very bright (>=220)
        assert all(c >= 220 for c in pixel[:3]), \
            f"Expected near-white dome pixel, got {tuple(pixel[:3])}"

    def test_led_color_in_spec(self):
        """Ensure board spec defines white and blue LED colors."""
        spec = load_spec(SPEC_PATH)
        led_colors = spec['component_defaults']['led_colors']
        assert 'blue' in led_colors, "spec must define a 'blue' LED color"
        assert 'white' in led_colors, "spec must define a 'white' LED color"


class TestArduino:
    def test_arduino_draws_pixels(self):
        img, grid, _ = _make_board()
        arr_before = np.array(img).copy()
        Arduino(start_col=10).draw(img, grid)
        arr_after = np.array(img)
        assert not np.array_equal(arr_before, arr_after), \
            "Arduino should change pixels"

    def test_arduino_body_dark_at_center(self):
        """The PCB body (dark navy) should appear over the center gap area."""
        img, grid, _ = _make_board()
        Arduino(start_col=10).draw(img, grid)
        arr = np.array(img)
        # Sample mid-body, midway between top and bottom pin rows
        # (avoid silkscreen letters which sit exactly at the center)
        x_first, _ = grid.hole_center('c', 10)
        x_last, _ = grid.hole_center('h', 10 + 17 - 1)
        body_cx = int(round((x_first + x_last) / 2))
        y_e, y_f = grid.center_gap_y_range()
        body_cy = int(round((y_e + y_f) / 2))
        # Step a bit off-center to dodge any silkscreen text
        sample_y = body_cy + int(round(grid.pitch_px * 0.6))
        pixel = arr[sample_y, body_cx]
        assert all(c < 100 for c in pixel[:3]), \
            f"Expected dark PCB pixel near center gap, got {tuple(pixel[:3])}"

    def test_arduino_straddles_center_gap(self):
        """The body must extend from above row e to below row f (across the gap)."""
        img, grid, _ = _make_board()
        Arduino(start_col=12).draw(img, grid)
        arr = np.array(img)
        x_mid, _ = grid.hole_center('c', 12 + 8)  # middle-ish column
        x_mid = int(round(x_mid))
        # Sample on row e and row f — both should be covered by the dark body
        _, y_e = grid.hole_center('e', 12 + 8)
        _, y_f = grid.hole_center('f', 12 + 8)
        for label, y in (('row e', int(round(y_e))), ('row f', int(round(y_f)))):
            pixel = arr[y, x_mid]
            # PCB body is much darker than the white board (255,255,255)
            assert max(pixel[:3]) < 150, \
                f"{label} not covered by Arduino body, got {tuple(pixel[:3])}"

    def test_arduino_pin_pads_visible(self):
        """Each pin position should have a light-colored pad over the dark body."""
        img, grid, spec = _make_board()
        Arduino(start_col=20).draw(img, grid)
        arr = np.array(img)
        pad_color = spec['component_defaults']['arduino']['pin_pad_color']
        # Sample several top-side pin centers — each should be near pad_color
        for col in (20, 23, 28):
            cx, cy = grid.hole_center('c', col)
            px, py = int(round(cx)), int(round(cy))
            pixel = arr[py, px]
            # Pad color is gold-ish (R > G > B); check ordering and brightness
            assert pixel[0] > pixel[2] and pixel[0] > 120, \
                f"Pin at col {col} not pad-coloured, got {tuple(pixel[:3])}"

    def test_arduino_usb_left(self):
        """USB connector should leave silver pixels protruding past the left edge."""
        img, grid, spec = _make_board()
        Arduino(start_col=10, usb_end='left').draw(img, grid)
        arr = np.array(img)
        # Sample slightly to the LEFT of col 10 at body-center y.
        # Body center y ≈ (row c + row h) / 2 = midway between rows e and f.
        y_e, y_f = grid.center_gap_y_range()
        body_cy = int(round((y_e + y_f) / 2))
        x_left, _ = grid.hole_center('c', 10)
        sample_x = int(round(x_left - grid.ppmm * 1.0))  # 1 mm left of col 10
        pixel = arr[body_cy, sample_x]
        # USB silver: bright + roughly balanced channels
        assert min(pixel[:3]) > 120 and max(pixel[:3]) - min(pixel[:3]) < 40, \
            f"Expected silver USB pixel left of col 10, got {tuple(pixel[:3])}"

    def test_arduino_usb_right(self):
        """usb_end='right' should put silver pixels past the right edge instead."""
        img, grid, _ = _make_board()
        Arduino(start_col=10, usb_end='right').draw(img, grid)
        arr = np.array(img)
        end_col = 10 + 17 - 1
        y_e, y_f = grid.center_gap_y_range()
        body_cy = int(round((y_e + y_f) / 2))
        x_right, _ = grid.hole_center('c', end_col)
        sample_x = int(round(x_right + grid.ppmm * 1.0))  # 1 mm right of last col
        pixel = arr[body_cy, sample_x]
        assert min(pixel[:3]) > 120 and max(pixel[:3]) - min(pixel[:3]) < 40, \
            f"Expected silver USB pixel right of last col, got {tuple(pixel[:3])}"

    def test_arduino_invalid_usb_end(self):
        try:
            Arduino(start_col=10, usb_end='top')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_arduino_invalid_top_row(self):
        img, grid, _ = _make_board()
        a = Arduino(start_col=10, top_row='f')  # f is bottom half
        try:
            a.draw(img, grid)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_arduino_invalid_bottom_row(self):
        img, grid, _ = _make_board()
        a = Arduino(start_col=10, bottom_row='c')  # c is top half
        try:
            a.draw(img, grid)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_arduino_out_of_bounds(self):
        img, grid, _ = _make_board()
        # 17 pins from col 60 → end_col = 76, way past 63
        a = Arduino(start_col=60)
        try:
            a.draw(img, grid)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_arduino_default_n_pins_from_spec(self):
        """Body span should match spec default_n_pins when not specified."""
        img, grid, spec = _make_board()
        n = spec['component_defaults']['arduino']['default_n_pins']
        Arduino(start_col=5).draw(img, grid)
        arr = np.array(img)
        # First pin position is dark/header-coloured
        cx, cy = grid.hole_center('c', 5)
        first = arr[int(round(cy)), int(round(cx))]
        assert max(first[:3]) < 220, \
            f"Expected pad/header at first pin, got {tuple(first[:3])}"
        # Last pin position (col 5 + n - 1) is also dark/header-coloured
        cx, cy = grid.hole_center('c', 5 + n - 1)
        last = arr[int(round(cy)), int(round(cx))]
        assert max(last[:3]) < 220, \
            f"Expected pad/header at last pin (col {5 + n - 1}), got {tuple(last[:3])}"
        # And the column one step beyond the body should still be plain board
        cx, cy = grid.hole_center('c', 5 + n + 1)
        beyond = arr[int(round(cy)), int(round(cx))]
        # That hole is rendered dark by holes.py — but the area between holes is white.
        # Sample halfway between cols to confirm we're past the body.
        cx_a, _ = grid.hole_center('c', 5 + n + 1)
        cx_b, _ = grid.hole_center('c', 5 + n + 2)
        mid_x = int(round((cx_a + cx_b) / 2))
        between = arr[int(round(cy)), mid_x]
        assert all(c > 200 for c in between[:3]), \
            f"Expected white board past Arduino end, got {tuple(between[:3])}"
