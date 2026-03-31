"""
Automated validation for rendered breadboard images.

Phase 1: validates blank board correctness (dimensions, holes, stripes, gap).
Replaces manual visual inspection with programmatic checks.
"""

import numpy as np
from PIL import Image
from generator.grid import BreadboardGrid


class BoardValidator:
    """Validates rendered board images against the spec."""

    def __init__(self, spec: dict, grid: BreadboardGrid):
        self.spec = spec
        self.grid = grid

    def validate_board(self, img: Image.Image) -> list[str]:
        """
        Validate a blank board image. Returns list of errors (empty = pass).

        Checks:
        - Image dimensions match spec
        - Holes are dark at expected positions
        - Power rail stripes have correct colors at expected positions
        - Center gap area is clear (no unexpected dark content)
        - Board body is white at non-feature positions
        """
        errors = []
        arr = np.array(img)

        errors.extend(self._check_dimensions(img))
        errors.extend(self._check_holes(arr))
        errors.extend(self._check_rail_stripes(arr))
        errors.extend(self._check_rail_grouping(arr))
        errors.extend(self._check_center_gap(arr))
        errors.extend(self._check_board_body(arr))

        return errors

    def _check_dimensions(self, img: Image.Image) -> list[str]:
        w, h = img.size
        ppmm = self.spec['board']['pixels_per_mm']
        exp_w = round(self.spec['board']['real_width_mm'] * ppmm)
        exp_h = round(self.spec['board']['real_height_mm'] * ppmm)
        errors = []
        if w != exp_w:
            errors.append(f"Width {w} != expected {exp_w}")
        if h != exp_h:
            errors.append(f"Height {h} != expected {exp_h}")
        return errors

    def _check_holes(self, arr: np.ndarray) -> list[str]:
        """Sample terminal and rail holes — pixel at center should be dark."""
        errors = []
        dark_threshold = 80  # max channel value to be considered "dark"

        # Sample ~10% of terminal holes
        terminal = self.grid.all_terminal_holes()
        sample_step = max(1, len(terminal) // 63)  # one per column roughly
        dark_count = 0
        sampled = 0

        for i in range(0, len(terminal), sample_step):
            row, col = terminal[i]
            cx, cy = self.grid.hole_center(row, col)
            px, py = int(round(cx)), int(round(cy))
            if 0 <= py < arr.shape[0] and 0 <= px < arr.shape[1]:
                pixel = arr[py, px]
                sampled += 1
                if all(c <= dark_threshold for c in pixel[:3]):
                    dark_count += 1

        if sampled > 0:
            ratio = dark_count / sampled
            if ratio < 0.85:
                errors.append(
                    f"Only {dark_count}/{sampled} ({ratio:.0%}) sampled terminal "
                    f"hole centers are dark (expected >85%)"
                )

        # Sample rail holes
        rail_holes = self.grid.all_rail_holes()
        sample_step = max(1, len(rail_holes) // 40)
        rail_dark = 0
        rail_sampled = 0

        for i in range(0, len(rail_holes), sample_step):
            row, col = rail_holes[i]
            cx, cy = self.grid.hole_center(row, col)
            px, py = int(round(cx)), int(round(cy))
            if 0 <= py < arr.shape[0] and 0 <= px < arr.shape[1]:
                pixel = arr[py, px]
                rail_sampled += 1
                if all(c <= dark_threshold for c in pixel[:3]):
                    rail_dark += 1

        if rail_sampled > 0:
            ratio = rail_dark / rail_sampled
            if ratio < 0.85:
                errors.append(
                    f"Only {rail_dark}/{rail_sampled} ({ratio:.0%}) sampled rail "
                    f"hole centers are dark (expected >85%)"
                )

        return errors

    def _check_rail_stripes(self, arr: np.ndarray) -> list[str]:
        """Check that rail stripe positions have the expected color.

        Stripes are on the outer edges of each 2-row rail block:
        - Red (+) on the board-edge side (outside)
        - Blue (-) on the terminal-area side (inside)
        """
        errors = []
        grid = self.grid
        half_pitch = grid.pitch_px / 2

        # Stripe y-positions must match board.py logic
        checks = [
            (grid.rail_y('p1+') - half_pitch, 'top_red', 0),
            (grid.rail_y('p1-') + half_pitch, 'top_blue', 2),
            (grid.rail_y('p2-') - half_pitch, 'bottom_blue', 2),
            (grid.rail_y('p2+') + half_pitch, 'bottom_red', 0),
        ]

        for stripe_y, label, dominant_channel in checks:
            # Sample between holes at terminal column 20 (mid-board)
            x = grid.col_x(20) + grid.pitch_px * 0.4
            px, py = int(round(x)), int(round(stripe_y))

            if 0 <= py < arr.shape[0] and 0 <= px < arr.shape[1]:
                pixel = arr[py, px]
                dominant_val = pixel[dominant_channel]
                other_channels = [pixel[c] for c in range(3) if c != dominant_channel]

                if dominant_val < 100 or dominant_val <= max(other_channels):
                    errors.append(
                        f"Rail stripe '{label}' color wrong at "
                        f"({px},{py}): pixel={tuple(pixel[:3])}, "
                        f"expected {'red' if dominant_channel == 0 else 'blue'} stripe"
                    )

        return errors

    def _check_rail_grouping(self, arr: np.ndarray) -> list[str]:
        """Check that gaps between rail groups are clear (no holes in gap columns)."""
        errors = []
        grid = self.grid
        groups = grid.rail_groups()

        # Check inter-group gap pixels are light (no holes)
        dark_threshold = 80
        gap_dark = 0
        gap_sampled = 0

        for i in range(len(groups) - 1):
            gap_col_start = groups[i][-1] + 1
            gap_col_end = groups[i + 1][0]
            for gc in range(gap_col_start, gap_col_end):
                if gc < 1 or gc > grid.n_cols:
                    continue
                x = grid.col_x(gc)
                y = grid.rail_y('p1+')
                px, py = int(round(x)), int(round(y))
                if 0 <= py < arr.shape[0] and 0 <= px < arr.shape[1]:
                    pixel = arr[py, px]
                    gap_sampled += 1
                    if all(c <= dark_threshold for c in pixel[:3]):
                        gap_dark += 1

        if gap_dark > 0:
            errors.append(
                f"Rail grouping: {gap_dark}/{gap_sampled} inter-group gap "
                f"pixels are dark (should be clear)"
            )

        return errors

    def _check_center_gap(self, arr: np.ndarray) -> list[str]:
        """Check that the center gap area has no dark hole pixels."""
        errors = []
        y_e, y_f = self.grid.center_gap_y_range()
        center_y = int(round((y_e + y_f) / 2))

        # Sample across the center line — should be light (no holes)
        dark_count = 0
        sampled = 0
        for col in range(1, self.grid.n_cols + 1, 5):
            x = int(round(self.grid.col_x(col)))
            if 0 <= center_y < arr.shape[0] and 0 <= x < arr.shape[1]:
                pixel = arr[center_y, x]
                sampled += 1
                if all(c < 100 for c in pixel[:3]):
                    dark_count += 1

        if dark_count > 0:
            errors.append(
                f"Center gap has {dark_count} dark pixels out of {sampled} "
                f"samples — should be clear"
            )

        return errors

    def _check_board_body(self, arr: np.ndarray) -> list[str]:
        """Check that non-feature areas of the board are white-ish."""
        errors = []
        grid = self.grid

        # Sample points between terminal rows (in the gap between rows a and b)
        _, y_a = grid.hole_center('a', 1)
        _, y_b = grid.hole_center('b', 1)
        mid_y = int(round((y_a + y_b) / 2))

        # Sample between columns (halfway between holes)
        light_count = 0
        sampled = 0
        for col in range(2, grid.n_cols, 4):
            x1 = grid.col_x(col)
            x2 = grid.col_x(col + 1)
            mid_x = int(round((x1 + x2) / 2))
            if 0 <= mid_y < arr.shape[0] and 0 <= mid_x < arr.shape[1]:
                pixel = arr[mid_y, mid_x]
                sampled += 1
                if all(c > 200 for c in pixel[:3]):
                    light_count += 1

        if sampled > 0:
            ratio = light_count / sampled
            if ratio < 0.9:
                errors.append(
                    f"Board body: only {light_count}/{sampled} ({ratio:.0%}) "
                    f"sampled inter-hole points are white (expected >90%)"
                )

        return errors


def validate_blank_board(spec: dict, img: Image.Image) -> list[str]:
    """Convenience function to validate a blank board image."""
    grid = BreadboardGrid(spec)
    validator = BoardValidator(spec, grid)
    return validator.validate_board(img)
