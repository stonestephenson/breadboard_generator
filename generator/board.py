"""
Board base renderer — draws the blank breadboard surface.

Draws: board body, rail stripes, rail +/- symbols, center gap channel, labels.
Does NOT draw holes — that's holes.py.
"""

import os
from PIL import Image, ImageDraw, ImageFont
from generator.grid import BreadboardGrid, load_spec
from generator.holes import draw_holes


def render_blank_board(spec_path: str, output_path: str) -> Image.Image:
    """
    Render a complete blank breadboard (board base + holes) and save to file.

    Renders at supersample resolution, then downscales for antialiasing.

    Args:
        spec_path: Path to board_spec.json.
        output_path: Path for output PNG.

    Returns:
        Final PIL Image at target resolution.
    """
    spec = load_spec(spec_path)
    ss = spec['rendering']['supersample_factor']
    base_ppmm = spec['board']['pixels_per_mm']
    render_ppmm = base_ppmm * ss

    grid = BreadboardGrid(spec, ppmm_override=render_ppmm)

    img = draw_board_base(grid)
    img = draw_holes(img, grid)

    # Downscale to target resolution
    if ss > 1:
        target_w = round(spec['board']['real_width_mm'] * base_ppmm)
        target_h = round(spec['board']['real_height_mm'] * base_ppmm)
        img = img.resize((target_w, target_h), Image.LANCZOS)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    return img


def draw_board_base(grid: BreadboardGrid) -> Image.Image:
    """
    Create board image and draw the base layer: body, stripes, gap, labels.

    Returns a new PIL Image sized to grid.board_size().
    """
    spec = grid.spec
    w, h = grid.board_size()

    bg_color = tuple(spec['rendering']['background_color'])
    img = Image.new('RGB', (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    # Board body — white rounded rectangle
    body_color = tuple(spec['board']['body_color'])
    ss_ratio = grid.ppmm / spec['board']['pixels_per_mm']
    border_r = int(spec['board']['border_radius_px'] * ss_ratio)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=border_r, fill=body_color)

    _draw_rail_stripes(draw, grid)
    _draw_center_gap(draw, grid)
    _draw_labels(draw, grid)
    _draw_rail_symbols(draw, grid)

    return img


# ── Internal drawing helpers ──────────────────────────────────────────


def _draw_rail_stripes(draw: ImageDraw.ImageDraw, grid: BreadboardGrid) -> None:
    """Draw thin continuous colored stripes on the outer edges of each rail section.

    On the real WB-102, each rail section is a 2-row block of grouped holes.
    The stripes run on the outer edges of this block — never on top of or
    between the hole rows:
    - Red (+) stripe on the board-edge side (outside)
    - Blue (-) stripe on the terminal-area side (inside)
    Stripes are continuous lines (no center gap) spanning the full rail width.
    """
    spec = grid.spec
    stripe_h = spec['power_rails']['stripe_width_mm'] * grid.ppmm
    pos_color = tuple(spec['power_rails']['positive_stripe_color'])
    neg_color = tuple(spec['power_rails']['negative_stripe_color'])

    w, _ = grid.board_size()
    margin = grid.ppmm * 1.5

    half_pitch = grid.pitch_px / 2

    # Top rail block: p1+ (outer row) and p1- (inner row)
    y_p1_plus = grid.rail_y('p1+')
    y_p1_minus = grid.rail_y('p1-')
    top_red_y = y_p1_plus - half_pitch      # red above outer row
    top_blue_y = y_p1_minus + half_pitch    # blue below inner row

    # Bottom rail block: p2- (inner row) and p2+ (outer row)
    y_p2_minus = grid.rail_y('p2-')
    y_p2_plus = grid.rail_y('p2+')
    bottom_blue_y = y_p2_minus - half_pitch  # blue above inner row
    bottom_red_y = y_p2_plus + half_pitch    # red below outer row

    stripes = [
        (top_red_y, pos_color),
        (top_blue_y, neg_color),
        (bottom_blue_y, neg_color),
        (bottom_red_y, pos_color),
    ]

    for y, color in stripes:
        draw.rectangle(
            [margin, y - stripe_h / 2, w - margin, y + stripe_h / 2],
            fill=color,
        )


def _draw_center_gap(draw: ImageDraw.ImageDraw, grid: BreadboardGrid) -> None:
    """Draw the center DIP channel as a narrow recessed groove."""
    w, _ = grid.board_size()
    y_e, y_f = grid.center_gap_y_range()
    center_y = (y_e + y_f) / 2

    # Narrow visible channel (~1mm wide) — subtle recessed look
    channel_half = grid.ppmm * 0.5
    gap_color = (240, 240, 237)  # very subtle — just slightly off-white
    draw.rectangle(
        [0, center_y - channel_half, w, center_y + channel_half],
        fill=gap_color,
    )

    # Thin edge lines for depth perception
    edge_color = (220, 220, 215)
    edge_h = max(1, grid.ppmm * 0.12)
    draw.rectangle(
        [0, center_y - channel_half, w, center_y - channel_half + edge_h],
        fill=edge_color,
    )
    draw.rectangle(
        [0, center_y + channel_half - edge_h, w, center_y + channel_half],
        fill=edge_color,
    )


def _get_font(size_px: float) -> ImageFont.FreeTypeFont:
    """Try to load a TrueType font, fall back to default."""
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


def _draw_labels(draw: ImageDraw.ImageDraw, grid: BreadboardGrid) -> None:
    """Draw row letters (a-j) at both ends and column numbers along edges."""
    spec = grid.spec
    label_color = tuple(spec['terminal_strip']['label_color'])
    font_size = spec['terminal_strip']['label_font_size_mm'] * grid.ppmm
    font = _get_font(font_size)

    rows_top = spec['terminal_strip']['row_labels_top']
    rows_bottom = spec['terminal_strip']['row_labels_bottom']
    all_rows = rows_top + rows_bottom

    # Row labels — positioned between inner rails and terminal area, at left/right ends
    # Left: halfway between board edge and column 1
    x_left = grid.col_x(1) / 2
    # Right: halfway between column 63 and board right edge
    board_w, _ = grid.board_size()
    x_right = (grid.col_x(grid.n_cols) + board_w) / 2

    for r in all_rows:
        y = grid.row_y(r)
        draw.text((x_left, y), r, fill=label_color, font=font, anchor='mm')
        draw.text((x_right, y), r, fill=label_color, font=font, anchor='mm')

    # Column number labels — between board edge and outermost rail
    col_labels = [1] + list(range(5, grid.n_cols, 5))
    if grid.n_cols not in col_labels:
        col_labels.append(grid.n_cols)

    # Top labels: between top edge and p1+ rail
    y_top = grid.rail_y('p1+') / 2
    # Bottom labels: between p2+ rail and bottom edge
    _, board_h = grid.board_size()
    y_bottom = (grid.rail_y('p2+') + board_h) / 2

    col_font = _get_font(font_size * 0.85)
    for c in col_labels:
        x = grid.col_x(c)
        draw.text((x, y_top), str(c), fill=label_color, font=col_font, anchor='mm')
        draw.text((x, y_bottom), str(c), fill=label_color, font=col_font, anchor='mm')


def _draw_rail_symbols(draw: ImageDraw.ImageDraw, grid: BreadboardGrid) -> None:
    """Draw red + and blue - symbols at both ends of each stripe line."""
    spec = grid.spec
    pos_color = tuple(spec['power_rails']['positive_stripe_color'])
    neg_color = tuple(spec['power_rails']['negative_stripe_color'])
    font_size = spec['terminal_strip']['label_font_size_mm'] * grid.ppmm
    font = _get_font(font_size * 1.1)

    w, _ = grid.board_size()
    margin = grid.ppmm * 1.5
    x_left = margin * 0.4
    x_right = w - margin * 0.4

    half_pitch = grid.pitch_px / 2

    # Must match stripe y-positions from _draw_rail_stripes
    symbols = [
        (grid.rail_y('p1+') - half_pitch, '+', pos_color),
        (grid.rail_y('p1-') + half_pitch, '-', neg_color),
        (grid.rail_y('p2-') - half_pitch, '-', neg_color),
        (grid.rail_y('p2+') + half_pitch, '+', pos_color),
    ]

    for y, symbol, color in symbols:
        draw.text((x_left, y), symbol, fill=color, font=font, anchor='mm')
        draw.text((x_right, y), symbol, fill=color, font=font, anchor='mm')
