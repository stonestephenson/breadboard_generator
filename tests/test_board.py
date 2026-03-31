"""Tests for board rendering and validation."""

import os
import pytest
from PIL import Image
from generator.grid import BreadboardGrid, load_spec
from generator.board import render_blank_board, draw_board_base
from generator.holes import draw_holes
from generator.validate import BoardValidator

SPEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'board_spec.json')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_board_test.png')


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


@pytest.fixture
def grid(spec):
    return BreadboardGrid(spec)


@pytest.fixture
def blank_board():
    return render_blank_board(SPEC_PATH, OUTPUT_PATH)


class TestRenderBlankBoard:
    def test_returns_image(self, blank_board):
        assert isinstance(blank_board, Image.Image)

    def test_output_file_created(self, blank_board):
        assert os.path.exists(OUTPUT_PATH)

    def test_correct_dimensions(self, blank_board, spec):
        ppmm = spec['board']['pixels_per_mm']
        exp_w = round(spec['board']['real_width_mm'] * ppmm)
        exp_h = round(spec['board']['real_height_mm'] * ppmm)
        assert blank_board.size == (exp_w, exp_h)

    def test_is_rgb(self, blank_board):
        assert blank_board.mode == 'RGB'


class TestBoardValidation:
    def test_blank_board_passes_validation(self, spec, blank_board):
        grid = BreadboardGrid(spec)
        validator = BoardValidator(spec, grid)
        errors = validator.validate_board(blank_board)
        assert errors == [], f"Validation errors: {errors}"

    def test_dimension_check_catches_wrong_size(self, spec, grid):
        wrong_img = Image.new('RGB', (100, 100), (255, 255, 255))
        validator = BoardValidator(spec, grid)
        errors = validator._check_dimensions(wrong_img)
        assert len(errors) == 2  # width and height both wrong

    def test_body_color_check(self, spec, grid):
        """All-black image should fail body color check."""
        import numpy as np
        w, h = grid.board_size()
        black_arr = np.zeros((h, w, 3), dtype=np.uint8)
        validator = BoardValidator(spec, grid)
        errors = validator._check_board_body(black_arr)
        assert len(errors) > 0


class TestSupersampling:
    def test_supersample_downscale(self, spec):
        """Rendered image should be at base ppmm, not supersample ppmm."""
        img = render_blank_board(SPEC_PATH, OUTPUT_PATH)
        base_ppmm = spec['board']['pixels_per_mm']
        exp_w = round(spec['board']['real_width_mm'] * base_ppmm)
        exp_h = round(spec['board']['real_height_mm'] * base_ppmm)
        assert img.size == (exp_w, exp_h)
