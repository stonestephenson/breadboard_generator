"""
Breadboard coordinate system — the ONLY module that does pixel math.

Converts logical positions (row, col) to pixel coordinates.
All other modules call this module for positions. Zero pixel math elsewhere.
"""

import json
from typing import Optional


def load_spec(spec_path: str) -> dict:
    """Load board spec from JSON file."""
    with open(spec_path) as f:
        return json.load(f)


class BreadboardGrid:
    """Coordinate system for a breadboard. Converts (row, col) to (px_x, px_y)."""

    def __init__(self, spec: dict, ppmm_override: Optional[float] = None):
        """
        Initialize from board_spec.json dict.

        Args:
            spec: Board specification dictionary.
            ppmm_override: Override pixels_per_mm (used for supersampling).
        """
        self.spec = spec
        self.ppmm = ppmm_override or spec['board']['pixels_per_mm']
        self.pitch_mm = spec['holes']['pitch_mm']
        self.pitch_px = self.pitch_mm * self.ppmm
        self.hole_diameter_mm = spec['holes']['diameter_mm']
        self.hole_diameter_px = self.hole_diameter_mm * self.ppmm

        self.board_width_mm = spec['board']['real_width_mm']
        self.board_height_mm = spec['board']['real_height_mm']
        self.board_width_px = round(self.board_width_mm * self.ppmm)
        self.board_height_px = round(self.board_height_mm * self.ppmm)

        self.n_cols = spec['terminal_strip']['columns']
        self.n_rows_per_half = spec['terminal_strip']['rows_per_half']
        self.center_gap_mm = spec['terminal_strip']['center_gap_mm']
        self.rows_top = list(spec['terminal_strip']['row_labels_top'])
        self.rows_bottom = list(spec['terminal_strip']['row_labels_bottom'])

        self.rail_offset_mm = spec['power_rails']['offset_from_terminal_mm']
        self.rail_spacing_mm = spec['power_rails']['rail_row_spacing_mm']
        self.holes_per_group = spec['power_rails']['holes_per_group']
        self.groups_per_segment = spec['power_rails']['groups_per_segment']
        self.inter_group_gap = spec['power_rails']['inter_group_gap_cols']
        self.center_gap_cols = spec['power_rails']['center_gap_cols']
        self.holes_per_segment = self.holes_per_group * self.groups_per_segment

        self._compute_positions()

    # ── Position computation ──────────────────────────────────────────

    def _compute_positions(self) -> None:
        self._compute_column_x()
        self._compute_row_y()
        self._compute_rail_columns()

    def _compute_column_x(self) -> None:
        """Compute x-center in mm for each terminal column (1..n_cols), centered on board."""
        total_width = (self.n_cols - 1) * self.pitch_mm
        x_start = (self.board_width_mm - total_width) / 2.0
        self._col_x_mm = {
            c: x_start + (c - 1) * self.pitch_mm
            for c in range(1, self.n_cols + 1)
        }

    def _compute_row_y(self) -> None:
        """Compute y-center in mm for all terminal rows and rail rows, centered vertically."""
        top_half_span = (self.n_rows_per_half - 1) * self.pitch_mm
        bottom_half_span = (self.n_rows_per_half - 1) * self.pitch_mm
        terminal_span = top_half_span + self.center_gap_mm + bottom_half_span

        total_span = (self.rail_spacing_mm + self.rail_offset_mm
                      + terminal_span
                      + self.rail_offset_mm + self.rail_spacing_mm)

        y_top = (self.board_height_mm - total_span) / 2.0

        # Rail positions (top: positive outside, negative inside)
        self._rail_y_mm = {
            'p1+': y_top,
            'p1-': y_top + self.rail_spacing_mm,
        }

        # Terminal rows top half (a-e)
        row_a_y = self._rail_y_mm['p1-'] + self.rail_offset_mm
        self._row_y_mm = {}
        for i, r in enumerate(self.rows_top):
            self._row_y_mm[r] = row_a_y + i * self.pitch_mm

        # Terminal rows bottom half (f-j)
        row_f_y = self._row_y_mm[self.rows_top[-1]] + self.center_gap_mm
        for i, r in enumerate(self.rows_bottom):
            self._row_y_mm[r] = row_f_y + i * self.pitch_mm

        # Rail positions (bottom: negative inside, positive outside)
        self._rail_y_mm['p2-'] = self._row_y_mm[self.rows_bottom[-1]] + self.rail_offset_mm
        self._rail_y_mm['p2+'] = self._rail_y_mm['p2-'] + self.rail_spacing_mm

    def _compute_rail_columns(self) -> None:
        """Compute rail hole columns: 10 groups of 5, with inter-group and center gaps.

        Layout (for 63 terminal columns):
          Segment 1: 5 groups of 5 holes with 1-col gaps between groups
          Center gap: 5 columns
          Segment 2: 5 groups of 5 holes with 1-col gaps between groups
        Rail holes align with terminal columns starting at column 1.
        """
        gpg = self.holes_per_group        # 5
        gps = self.groups_per_segment     # 5
        ig = self.inter_group_gap         # 1
        cg = self.center_gap_cols         # 5
        group_stride = gpg + ig           # 6 cols per group slot

        self._rail_groups: list[list[int]] = []
        self._rail_seg1_cols: list[int] = []
        self._rail_seg2_cols: list[int] = []

        # Segment 1 starts at terminal column 1
        col = 1
        for _ in range(gps):
            group = list(range(col, col + gpg))
            self._rail_groups.append(group)
            self._rail_seg1_cols.extend(group)
            col += group_stride

        # Jump over center gap (col already advanced by one inter-group gap
        # past the last group, so add the remaining center gap)
        col += cg - ig

        # Segment 2
        for _ in range(gps):
            group = list(range(col, col + gpg))
            self._rail_groups.append(group)
            self._rail_seg2_cols.extend(group)
            col += group_stride

        self._rail_all_cols = self._rail_seg1_cols + self._rail_seg2_cols

    # ── Public API ────────────────────────────────────────────────────

    def hole_center(self, row: str, col: int) -> tuple[float, float]:
        """
        Convert logical position to pixel center.

        Args:
            row: 'a'-'j' for terminal, 'p1+', 'p1-', 'p2+', 'p2-' for rails.
            col: 1-63 for terminal columns, rail column index for rails.

        Returns:
            (x_px, y_px) center of that hole.
        """
        x_mm = self._col_x_mm[col]

        if row in self._row_y_mm:
            y_mm = self._row_y_mm[row]
        elif row in self._rail_y_mm:
            y_mm = self._rail_y_mm[row]
        else:
            raise ValueError(f"Unknown row: {row!r}")

        return (x_mm * self.ppmm, y_mm * self.ppmm)

    def hole_rect(self, row: str, col: int) -> tuple[float, float, float, float]:
        """Returns (x, y, w, h) bounding box of hole."""
        cx, cy = self.hole_center(row, col)
        d = self.hole_diameter_px
        return (cx - d / 2, cy - d / 2, d, d)

    def board_size(self) -> tuple[int, int]:
        """Returns (width_px, height_px) of full board."""
        return (self.board_width_px, self.board_height_px)

    def all_terminal_holes(self) -> list[tuple[str, int]]:
        """Returns list of all (row, col) terminal hole positions."""
        rows = self.rows_top + self.rows_bottom
        return [(r, c) for r in rows for c in range(1, self.n_cols + 1)]

    def all_rail_holes(self) -> list[tuple[str, int]]:
        """Returns list of all (rail_id, col) power rail hole positions."""
        rail_ids = ['p1+', 'p1-', 'p2-', 'p2+']
        return [(r, c) for r in rail_ids for c in self._rail_all_cols]

    def rail_columns(self) -> tuple[list[int], list[int]]:
        """Returns (segment1_cols, segment2_cols) for rail hole positions."""
        return (list(self._rail_seg1_cols), list(self._rail_seg2_cols))

    def rail_groups(self) -> list[list[int]]:
        """Returns list of 10 groups, each a list of 5 column numbers."""
        return [list(g) for g in self._rail_groups]

    def rail_y(self, rail_id: str) -> float:
        """Returns y pixel position for a rail row."""
        return self._rail_y_mm[rail_id] * self.ppmm

    def row_y(self, row: str) -> float:
        """Returns y pixel position for a terminal row."""
        return self._row_y_mm[row] * self.ppmm

    def col_x(self, col: int) -> float:
        """Returns x pixel position for a column."""
        return self._col_x_mm[col] * self.ppmm

    def center_gap_y_range(self) -> tuple[float, float]:
        """Returns (y_top_px, y_bottom_px) — y of row e center and row f center."""
        e_y = self._row_y_mm[self.rows_top[-1]]
        f_y = self._row_y_mm[self.rows_bottom[0]]
        return (e_y * self.ppmm, f_y * self.ppmm)

    def connected_holes(self, row: str, col: int) -> list[tuple[str, int]]:
        """
        Returns all holes electrically connected to this one.

        Terminal holes: same column, same half (a-e or f-j).
        Rail holes: same rail, same segment only.
        """
        if row in self._rail_y_mm:
            if col in self._rail_seg1_cols:
                return [(row, c) for c in self._rail_seg1_cols if c != col]
            elif col in self._rail_seg2_cols:
                return [(row, c) for c in self._rail_seg2_cols if c != col]
            return []

        if row in self.rows_top:
            return [(r, col) for r in self.rows_top if r != row]
        elif row in self.rows_bottom:
            return [(r, col) for r in self.rows_bottom if r != row]

        raise ValueError(f"Unknown row: {row!r}")
