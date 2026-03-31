"""Unit tests for generator/components.py and generator/wires.py."""

import os
import numpy as np
from PIL import Image
from generator.grid import BreadboardGrid, load_spec
from generator.board import draw_board_base
from generator.holes import draw_holes
from generator.wires import draw_wire
from generator.components import Resistor, LED

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
