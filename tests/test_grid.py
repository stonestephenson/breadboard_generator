"""Unit tests for generator/grid.py — the coordinate system."""

import os
import pytest
from generator.grid import BreadboardGrid, load_spec

SPEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'board_spec.json')


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


@pytest.fixture
def grid(spec):
    return BreadboardGrid(spec)


# ── Board dimensions ──────────────────────────────────────────────────


class TestBoardDimensions:
    def test_board_size_matches_spec(self, grid, spec):
        w, h = grid.board_size()
        ppmm = spec['board']['pixels_per_mm']
        assert w == round(spec['board']['real_width_mm'] * ppmm)
        assert h == round(spec['board']['real_height_mm'] * ppmm)

    def test_board_is_landscape(self, grid):
        w, h = grid.board_size()
        assert w > h, "Board should be landscape (wider than tall)"

    def test_board_dimensions_at_4ppmm(self, grid):
        w, h = grid.board_size()
        # 165.1mm * 4 = 660.4 → 660, 54.0mm * 4 = 216.0
        assert w == 660
        assert h == 216


# ── Hole spacing ──────────────────────────────────────────────────────


class TestHoleSpacing:
    def test_horizontal_pitch_all_columns(self, grid, spec):
        expected = spec['holes']['pitch_mm'] * spec['board']['pixels_per_mm']
        for col in range(1, 63):
            x1, _ = grid.hole_center('a', col)
            x2, _ = grid.hole_center('a', col + 1)
            assert abs((x2 - x1) - expected) < 0.01, f"Col {col}→{col+1} pitch off"

    def test_vertical_pitch_top_half(self, grid, spec):
        expected = spec['holes']['pitch_mm'] * spec['board']['pixels_per_mm']
        rows = spec['terminal_strip']['row_labels_top']
        for i in range(len(rows) - 1):
            _, y1 = grid.hole_center(rows[i], 1)
            _, y2 = grid.hole_center(rows[i + 1], 1)
            assert abs((y2 - y1) - expected) < 0.01, f"Row {rows[i]}→{rows[i+1]} pitch off"

    def test_vertical_pitch_bottom_half(self, grid, spec):
        expected = spec['holes']['pitch_mm'] * spec['board']['pixels_per_mm']
        rows = spec['terminal_strip']['row_labels_bottom']
        for i in range(len(rows) - 1):
            _, y1 = grid.hole_center(rows[i], 1)
            _, y2 = grid.hole_center(rows[i + 1], 1)
            assert abs((y2 - y1) - expected) < 0.01, f"Row {rows[i]}→{rows[i+1]} pitch off"

    def test_center_gap(self, grid, spec):
        expected = spec['terminal_strip']['center_gap_mm'] * spec['board']['pixels_per_mm']
        _, ye = grid.hole_center('e', 1)
        _, yf = grid.hole_center('f', 1)
        assert abs((yf - ye) - expected) < 0.01, "Center gap size mismatch"

    def test_pitch_is_consistent_across_rows(self, grid):
        """Pitch should be the same regardless of which row we measure."""
        x1_a, _ = grid.hole_center('a', 10)
        x2_a, _ = grid.hole_center('a', 11)
        x1_j, _ = grid.hole_center('j', 10)
        x2_j, _ = grid.hole_center('j', 11)
        assert abs((x2_a - x1_a) - (x2_j - x1_j)) < 0.001


# ── Symmetry ──────────────────────────────────────────────────────────


class TestSymmetry:
    def test_vertical_symmetry_rails(self, grid):
        _, h = grid.board_size()
        center = h / 2.0
        _, y_p1_plus = grid.hole_center('p1+', 10)
        _, y_p2_plus = grid.hole_center('p2+', 10)
        assert abs((center - y_p1_plus) - (y_p2_plus - center)) < 0.1

    def test_vertical_symmetry_terminal(self, grid):
        _, h = grid.board_size()
        center = h / 2.0
        _, y_a = grid.hole_center('a', 1)
        _, y_j = grid.hole_center('j', 1)
        assert abs((center - y_a) - (y_j - center)) < 0.1

    def test_horizontal_centering(self, grid):
        w, _ = grid.board_size()
        x1, _ = grid.hole_center('a', 1)
        x63, _ = grid.hole_center('a', 63)
        left_margin = x1
        right_margin = w - x63
        assert abs(left_margin - right_margin) < 0.5, "Columns not centered (sub-pixel ok)"

    def test_all_holes_within_board(self, grid):
        w, h = grid.board_size()
        for row, col in grid.all_terminal_holes() + grid.all_rail_holes():
            x, y = grid.hole_center(row, col)
            assert 0 < x < w, f"Hole ({row},{col}) x={x} outside board width {w}"
            assert 0 < y < h, f"Hole ({row},{col}) y={y} outside board height {h}"


# ── Hole counts ───────────────────────────────────────────────────────


class TestHoleCounts:
    def test_terminal_hole_count(self, grid):
        assert len(grid.all_terminal_holes()) == 630

    def test_rail_hole_count(self, grid):
        assert len(grid.all_rail_holes()) == 200

    def test_total_tie_points(self, grid):
        total = len(grid.all_terminal_holes()) + len(grid.all_rail_holes())
        assert total == 830

    def test_no_duplicate_terminal_holes(self, grid):
        holes = grid.all_terminal_holes()
        assert len(holes) == len(set(holes))

    def test_no_duplicate_rail_holes(self, grid):
        holes = grid.all_rail_holes()
        assert len(holes) == len(set(holes))


# ── Rail layout ───────────────────────────────────────────────────────


class TestRailLayout:
    def test_segment_sizes(self, grid, spec):
        seg1, seg2 = grid.rail_columns()
        hpg = spec['power_rails']['holes_per_group']
        gps = spec['power_rails']['groups_per_segment']
        assert len(seg1) == hpg * gps
        assert len(seg2) == hpg * gps

    def test_center_gap_between_segments(self, grid, spec):
        seg1, seg2 = grid.rail_columns()
        actual_gap = seg2[0] - seg1[-1] - 1
        assert actual_gap == spec['power_rails']['center_gap_cols']

    def test_group_count(self, grid, spec):
        groups = grid.rail_groups()
        total_groups = spec['power_rails']['groups_per_segment'] * 2
        assert len(groups) == total_groups

    def test_group_sizes(self, grid, spec):
        groups = grid.rail_groups()
        hpg = spec['power_rails']['holes_per_group']
        for i, g in enumerate(groups):
            assert len(g) == hpg, f"Group {i} has {len(g)} holes, expected {hpg}"

    def test_inter_group_gaps(self, grid, spec):
        """Adjacent groups within a segment should have the specified gap."""
        groups = grid.rail_groups()
        gps = spec['power_rails']['groups_per_segment']
        ig = spec['power_rails']['inter_group_gap_cols']
        # Check within segment 1 (groups 0..4) and segment 2 (groups 5..9)
        for seg_start in [0, gps]:
            for i in range(seg_start, seg_start + gps - 1):
                gap = groups[i + 1][0] - groups[i][-1] - 1
                assert gap == ig, f"Gap between group {i} and {i+1} is {gap}, expected {ig}"

    def test_rails_start_at_configured_column(self, grid, spec):
        """Rail groups should start at the configured rail_start_col."""
        groups = grid.rail_groups()
        expected_start = spec['power_rails'].get('rail_start_col', 1)
        assert groups[0][0] == expected_start

    def test_top_rail_spacing(self, grid, spec):
        expected = spec['power_rails']['rail_row_spacing_mm'] * spec['board']['pixels_per_mm']
        _, y_plus = grid.hole_center('p1+', 10)
        _, y_minus = grid.hole_center('p1-', 10)
        assert abs((y_minus - y_plus) - expected) < 0.01

    def test_bottom_rail_spacing(self, grid, spec):
        expected = spec['power_rails']['rail_row_spacing_mm'] * spec['board']['pixels_per_mm']
        _, y_minus = grid.hole_center('p2-', 10)
        _, y_plus = grid.hole_center('p2+', 10)
        assert abs((y_plus - y_minus) - expected) < 0.01

    def test_rail_offset_from_terminal(self, grid, spec):
        """Distance from inner rail to nearest terminal row should match spec."""
        expected = spec['power_rails']['offset_from_terminal_mm'] * spec['board']['pixels_per_mm']
        col = 10
        _, y_p1_minus = grid.hole_center('p1-', col)
        _, y_a = grid.hole_center('a', col)
        assert abs((y_a - y_p1_minus) - expected) < 0.01

        _, y_j = grid.hole_center('j', col)
        _, y_p2_minus = grid.hole_center('p2-', col)
        assert abs((y_p2_minus - y_j) - expected) < 0.01

    def test_rail_order_top_outside_positive(self, grid):
        """p1+ should be above p1- (positive on outside)."""
        _, y_plus = grid.hole_center('p1+', 10)
        _, y_minus = grid.hole_center('p1-', 10)
        assert y_plus < y_minus

    def test_rail_order_bottom_outside_positive(self, grid):
        """p2+ should be below p2- (positive on outside)."""
        _, y_minus = grid.hole_center('p2-', 10)
        _, y_plus = grid.hole_center('p2+', 10)
        assert y_plus > y_minus


# ── Connectivity ──────────────────────────────────────────────────────


class TestConnectivity:
    def test_terminal_top_half(self, grid):
        connected = grid.connected_holes('a', 1)
        expected = [('b', 1), ('c', 1), ('d', 1), ('e', 1)]
        assert sorted(connected) == sorted(expected)

    def test_terminal_bottom_half(self, grid):
        connected = grid.connected_holes('f', 1)
        expected = [('g', 1), ('h', 1), ('i', 1), ('j', 1)]
        assert sorted(connected) == sorted(expected)

    def test_no_cross_gap_connectivity(self, grid):
        connected = grid.connected_holes('a', 1)
        bottom_rows = set(grid.rows_bottom)
        for r, _ in connected:
            assert r not in bottom_rows

    def test_terminal_connected_count(self, grid):
        # Each terminal hole connects to 4 others (same col, same half)
        assert len(grid.connected_holes('c', 30)) == 4

    def test_rail_same_segment(self, grid):
        seg1, seg2 = grid.rail_columns()
        connected = grid.connected_holes('p1+', seg1[0])
        assert len(connected) == len(seg1) - 1
        for _, c in connected:
            assert c in seg1

    def test_rail_cross_segment_not_connected(self, grid):
        seg1, seg2 = grid.rail_columns()
        connected = grid.connected_holes('p1+', seg1[0])
        connected_cols = {c for _, c in connected}
        assert not connected_cols.intersection(seg2)

    def test_rail_segment2_connectivity(self, grid):
        seg1, seg2 = grid.rail_columns()
        connected = grid.connected_holes('p2-', seg2[5])
        assert len(connected) == len(seg2) - 1
        for _, c in connected:
            assert c in seg2


# ── Hole rect ─────────────────────────────────────────────────────────


class TestHoleRect:
    def test_centered_on_hole_center(self, grid):
        cx, cy = grid.hole_center('a', 1)
        x, y, w, h = grid.hole_rect('a', 1)
        assert abs((x + w / 2) - cx) < 0.01
        assert abs((y + h / 2) - cy) < 0.01

    def test_size_matches_spec(self, grid, spec):
        _, _, w, h = grid.hole_rect('a', 1)
        expected = spec['holes']['diameter_mm'] * spec['board']['pixels_per_mm']
        assert abs(w - expected) < 0.01
        assert abs(h - expected) < 0.01


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_invalid_row_raises(self, grid):
        with pytest.raises(ValueError):
            grid.hole_center('z', 1)

    def test_invalid_row_connected_raises(self, grid):
        with pytest.raises(ValueError):
            grid.connected_holes('z', 1)

    def test_ppmm_override_scales_positions(self, spec):
        grid_1x = BreadboardGrid(spec)
        grid_2x = BreadboardGrid(spec, ppmm_override=8.0)

        cx1, cy1 = grid_1x.hole_center('a', 1)
        cx2, cy2 = grid_2x.hole_center('a', 1)

        assert abs(cx2 / cx1 - 2.0) < 0.01
        assert abs(cy2 / cy1 - 2.0) < 0.01

    def test_ppmm_override_scales_board_size(self, spec):
        grid_1x = BreadboardGrid(spec)
        grid_2x = BreadboardGrid(spec, ppmm_override=8.0)

        w1, h1 = grid_1x.board_size()
        w2, h2 = grid_2x.board_size()

        assert abs(w2 / w1 - 2.0) < 0.05
        assert abs(h2 / h1 - 2.0) < 0.05

    def test_center_gap_y_range(self, grid):
        y_top, y_bottom = grid.center_gap_y_range()
        assert y_bottom > y_top
        # Should match e and f y positions
        _, ye = grid.hole_center('e', 1)
        _, yf = grid.hole_center('f', 1)
        assert abs(y_top - ye) < 0.01
        assert abs(y_bottom - yf) < 0.01
