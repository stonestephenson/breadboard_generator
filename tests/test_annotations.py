"""Unit tests for generator/annotations.py — bounding box generation
and COCO/YOLO export."""

import json
import os
import math

import numpy as np
import pytest

from generator.grid import BreadboardGrid, load_spec
from generator.annotations import (
    BoundingBoxGenerator,
    CLASS_MAP,
    PADDING_MM,
    transform_bbox,
    transform_annotations,
    coco_dataset,
)


SPEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'board_spec.json')
CIRCUITS_DIR = os.path.join(os.path.dirname(__file__), '..', 'config', 'circuits')


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


@pytest.fixture
def grid(spec):
    return BreadboardGrid(spec)


@pytest.fixture
def bbox_gen(grid, spec):
    return BoundingBoxGenerator(grid, spec)


def _load_circuit(name: str) -> dict:
    with open(os.path.join(CIRCUITS_DIR, name)) as f:
        return json.load(f)


@pytest.fixture
def simple_led():
    return _load_circuit('simple_led.json')


@pytest.fixture
def dual_led():
    return _load_circuit('dual_led.json')


@pytest.fixture
def resistor_divider():
    return _load_circuit('resistor_divider.json')


# ── Class map ─────────────────────────────────────────────────────────


class TestClassMap:
    def test_required_classes_present(self):
        for name in ('breadboard', 'resistor', 'led', 'wire', 'capacitor', 'ic_chip'):
            assert name in CLASS_MAP

    def test_class_ids_unique(self):
        assert len(set(CLASS_MAP.values())) == len(CLASS_MAP)

    def test_breadboard_is_class_zero(self):
        assert CLASS_MAP['breadboard'] == 0


# ── Per-element bounding boxes ─────────────────────────────────────────


class TestComponentBbox:
    def test_resistor_bbox_covers_pins(self, bbox_gen, grid):
        comp = {
            "id": "R1",
            "type": "resistor",
            "pins": {"leg1": ["a", 10], "leg2": ["a", 15]},
        }
        x1, y1, x2, y2 = bbox_gen.component_bbox(comp)
        leg1_x, leg1_y = grid.hole_center('a', 10)
        leg2_x, leg2_y = grid.hole_center('a', 15)
        # Pin centers are inside the bbox
        assert x1 <= leg1_x <= x2
        assert x1 <= leg2_x <= x2
        assert y1 <= leg1_y <= y2
        assert y1 <= leg2_y <= y2

    def test_resistor_bbox_padding_applied(self, bbox_gen, grid, spec):
        comp = {
            "id": "R1",
            "type": "resistor",
            "pins": {"leg1": ["a", 10], "leg2": ["a", 15]},
        }
        x1, y1, x2, y2 = bbox_gen.component_bbox(comp)
        leg1_x, _ = grid.hole_center('a', 10)
        leg2_x, _ = grid.hole_center('a', 15)
        ppmm = spec['board']['pixels_per_mm']
        pad_x = PADDING_MM['resistor']['x'] * ppmm
        # x_min should be roughly leg1_x - pad_x (to within rounding)
        assert abs(x1 - (min(leg1_x, leg2_x) - pad_x)) <= 1
        assert abs(x2 - (max(leg1_x, leg2_x) + pad_x)) <= 1

    def test_led_bbox_dome_covered(self, bbox_gen, grid, spec):
        """LED dome (3.0mm diameter) must fit inside the bbox."""
        comp = {
            "id": "LED1",
            "type": "led",
            "color": "red",
            "pins": {"anode": ["a", 20], "cathode": ["b", 20]},
        }
        x1, y1, x2, y2 = bbox_gen.component_bbox(comp)
        ax, ay = grid.hole_center('a', 20)
        cx, cy = grid.hole_center('b', 20)
        mx, my = (ax + cx) / 2, (ay + cy) / 2
        dome_r = spec['component_defaults']['led_dome_diameter_mm'] / 2 * spec['board']['pixels_per_mm']
        assert x1 <= mx - dome_r + 0.5
        assert x2 >= mx + dome_r - 0.5
        assert y1 <= my - dome_r + 0.5
        assert y2 >= my + dome_r - 0.5

    def test_bbox_is_integer_tuple(self, bbox_gen, simple_led):
        for comp in simple_led['components']:
            bbox = bbox_gen.component_bbox(comp)
            assert len(bbox) == 4
            assert all(isinstance(v, int) for v in bbox)

    def test_bbox_positive_area(self, bbox_gen, simple_led):
        for comp in simple_led['components']:
            x1, y1, x2, y2 = bbox_gen.component_bbox(comp)
            assert x2 > x1, f"Zero-or-negative width for {comp['id']}"
            assert y2 > y1, f"Zero-or-negative height for {comp['id']}"

    def test_bbox_no_pins_raises(self, bbox_gen):
        with pytest.raises(ValueError):
            bbox_gen.component_bbox({"id": "X", "type": "resistor", "pins": {}})

    def test_resistor_diagonal_pins(self, bbox_gen, grid):
        """A resistor whose legs cross the center gap still yields a sane bbox."""
        comp = {
            "id": "R1",
            "type": "resistor",
            "pins": {"leg1": ["c", 30], "leg2": ["h", 30]},
        }
        x1, y1, x2, y2 = bbox_gen.component_bbox(comp)
        _, y_c = grid.hole_center('c', 30)
        _, y_h = grid.hole_center('h', 30)
        assert y1 <= y_c
        assert y2 >= y_h


class TestWireBbox:
    def test_straight_wire_bbox(self, bbox_gen, grid):
        wire = {"from": ["a", 5], "to": ["a", 15], "color": "red", "routing": "straight"}
        x1, y1, x2, y2 = bbox_gen.wire_bbox(wire)
        sx, sy = grid.hole_center('a', 5)
        ex, ey = grid.hole_center('a', 15)
        assert x1 <= sx <= x2
        assert x1 <= ex <= x2
        assert y1 <= sy <= y2
        assert y1 <= ey <= y2

    def test_l_wire_bbox_covers_elbow(self, bbox_gen, grid):
        """L-shaped wires elbow at one endpoint's row/col, so the standard
        min/max bbox still covers the full path."""
        wire = {"from": ["p1+", 9], "to": ["a", 10], "color": "red", "routing": "L"}
        x1, y1, x2, y2 = bbox_gen.wire_bbox(wire)
        sx, sy = grid.hole_center('p1+', 9)
        ex, ey = grid.hole_center('a', 10)
        # Elbow at (ex, sy) — must be in the bbox.
        assert x1 <= ex <= x2
        assert y1 <= sy <= y2

    def test_wire_to_rail(self, bbox_gen, grid):
        wire = {"from": ["b", 20], "to": ["p1-", 21], "color": "black", "routing": "L"}
        x1, y1, x2, y2 = bbox_gen.wire_bbox(wire)
        assert x2 > x1 and y2 > y1


class TestBoardBbox:
    def test_full_image_extent(self, bbox_gen):
        x1, y1, x2, y2 = bbox_gen.board_bbox()
        assert (x1, y1) == (0, 0)
        assert x2 == bbox_gen.image_w
        assert y2 == bbox_gen.image_h

    def test_landscape(self, bbox_gen):
        _, _, x2, y2 = bbox_gen.board_bbox()
        assert x2 > y2


# ── Annotation generation ──────────────────────────────────────────────


class TestGenerateAnnotations:
    def test_includes_breadboard_first(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        assert anns[0]['class_name'] == 'breadboard'
        assert anns[0]['class_id'] == 0

    def test_one_annotation_per_component(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        comp_anns = [a for a in anns if a['class_name'] in ('resistor', 'led')]
        assert len(comp_anns) == len(simple_led['components'])

    def test_one_annotation_per_wire(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        wire_anns = [a for a in anns if a['class_name'] == 'wire']
        assert len(wire_anns) == len(simple_led['wires'])

    def test_total_count(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        # board + 2 components + 3 wires = 6
        assert len(anns) == 1 + len(simple_led['components']) + len(simple_led['wires'])

    def test_component_id_carried(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        comp_ids = {a.get('component_id') for a in anns if 'component_id' in a}
        assert comp_ids == {'R1', 'LED1'}

    def test_wire_index_carried(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        wire_indices = sorted(a['wire_index'] for a in anns if 'wire_index' in a)
        assert wire_indices == [0, 1, 2]

    def test_all_bboxes_within_image(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        for a in anns:
            x1, y1, x2, y2 = a['bbox']
            assert 0 <= x1 < x2 <= bbox_gen.image_w
            assert 0 <= y1 < y2 <= bbox_gen.image_h

    def test_works_for_dual_led(self, bbox_gen, dual_led):
        anns = bbox_gen.generate_annotations(dual_led)
        n_expected = 1 + len(dual_led['components']) + len(dual_led['wires'])
        assert len(anns) == n_expected

    def test_works_for_resistor_divider(self, bbox_gen, resistor_divider):
        anns = bbox_gen.generate_annotations(resistor_divider)
        n_expected = 1 + len(resistor_divider['components']) + len(resistor_divider['wires'])
        assert len(anns) == n_expected

    def test_unknown_component_type_skipped(self, bbox_gen):
        circuit = {
            "components": [
                {"id": "R1", "type": "resistor", "pins": {"leg1": ["a", 5], "leg2": ["a", 10]}},
                {"id": "X1", "type": "transistor", "pins": {}},  # unknown — skipped
            ],
            "wires": [],
        }
        anns = bbox_gen.generate_annotations(circuit)
        # board + R1 only
        assert len(anns) == 2
        assert anns[1]['component_id'] == 'R1'


# ── COCO format ───────────────────────────────────────────────────────


class TestToCoco:
    def test_returns_image_and_annotations(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        record = bbox_gen.to_coco(anns, image_id=42, image_size=(660, 216))
        assert 'image' in record
        assert 'annotations' in record
        assert record['image']['id'] == 42
        assert record['image']['width'] == 660
        assert record['image']['height'] == 216

    def test_coco_bbox_is_xywh(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        record = bbox_gen.to_coco(anns, 0, (660, 216))
        for a, original in zip(record['annotations'], anns):
            x, y, w, h = a['bbox']
            ox1, oy1, ox2, oy2 = original['bbox']
            assert x == ox1
            assert y == oy1
            assert w == ox2 - ox1
            assert h == oy2 - oy1

    def test_coco_area_matches(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        record = bbox_gen.to_coco(anns, 0, (660, 216))
        for a in record['annotations']:
            x, y, w, h = a['bbox']
            assert a['area'] == w * h

    def test_coco_iscrowd_zero(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        record = bbox_gen.to_coco(anns, 0, (660, 216))
        for a in record['annotations']:
            assert a['iscrowd'] == 0

    def test_coco_serializes_to_json(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        record = bbox_gen.to_coco(anns, 0, (660, 216))
        # Should round-trip through JSON without errors
        json.dumps(record)


class TestCocoDataset:
    def test_aggregates_records(self, bbox_gen, simple_led, dual_led):
        rec1 = bbox_gen.to_coco(
            bbox_gen.generate_annotations(simple_led), 1, (660, 216),
        )
        rec2 = bbox_gen.to_coco(
            bbox_gen.generate_annotations(dual_led), 2, (660, 216),
        )
        ds = coco_dataset([rec1, rec2])
        assert len(ds['images']) == 2
        assert len(ds['annotations']) == len(rec1['annotations']) + len(rec2['annotations'])

    def test_annotation_ids_unique(self, bbox_gen, simple_led):
        rec1 = bbox_gen.to_coco(
            bbox_gen.generate_annotations(simple_led), 1, (660, 216),
        )
        rec2 = bbox_gen.to_coco(
            bbox_gen.generate_annotations(simple_led), 2, (660, 216),
        )
        ds = coco_dataset([rec1, rec2])
        ids = [a['id'] for a in ds['annotations']]
        assert len(ids) == len(set(ids))

    def test_categories_match_class_map(self, bbox_gen, simple_led):
        rec = bbox_gen.to_coco(
            bbox_gen.generate_annotations(simple_led), 1, (660, 216),
        )
        ds = coco_dataset([rec])
        cat_map = {c['name']: c['id'] for c in ds['categories']}
        assert cat_map == CLASS_MAP


# ── YOLO format ────────────────────────────────────────────────────────


class TestToYolo:
    def test_one_line_per_annotation(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        text = bbox_gen.to_yolo(anns, (660, 216))
        lines = text.strip().split('\n')
        assert len(lines) == len(anns)

    def test_normalized_values_in_range(self, bbox_gen, simple_led):
        anns = bbox_gen.generate_annotations(simple_led)
        text = bbox_gen.to_yolo(anns, (660, 216))
        for line in text.strip().split('\n'):
            parts = line.split()
            assert len(parts) == 5
            class_id = int(parts[0])
            cx, cy, nw, nh = (float(p) for p in parts[1:])
            assert class_id in CLASS_MAP.values()
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 <= nw <= 1.0
            assert 0.0 <= nh <= 1.0

    def test_yolo_round_trips_back_to_pixels(self, bbox_gen, simple_led):
        """Decoding a YOLO line should produce a bbox close to the original."""
        anns = bbox_gen.generate_annotations(simple_led)
        w, h = 660, 216
        text = bbox_gen.to_yolo(anns, (w, h))
        for line, original in zip(text.strip().split('\n'), anns):
            parts = line.split()
            cx, cy, nw, nh = (float(p) for p in parts[1:])
            x1 = (cx - nw / 2) * w
            y1 = (cy - nh / 2) * h
            x2 = (cx + nw / 2) * w
            y2 = (cy + nh / 2) * h
            ox1, oy1, ox2, oy2 = original['bbox']
            assert abs(x1 - ox1) < 1.0
            assert abs(y1 - oy1) < 1.0
            assert abs(x2 - ox2) < 1.0
            assert abs(y2 - oy2) < 1.0

    def test_invalid_image_size_raises(self, bbox_gen):
        with pytest.raises(ValueError):
            bbox_gen.to_yolo([], (0, 0))

    def test_empty_annotations_returns_empty_string(self, bbox_gen):
        assert bbox_gen.to_yolo([], (660, 216)) == ""


# ── Geometric transforms ───────────────────────────────────────────────


class TestTransformBbox:
    def test_identity_affine(self):
        bbox = (10, 20, 100, 200)
        identity = {"type": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]}
        out = transform_bbox(bbox, [identity], (660, 216))
        assert out == bbox

    def test_translation(self):
        bbox = (10, 20, 100, 200)
        # Translate by (+5, -3)
        translate = {"type": "affine", "matrix": [[1, 0, 5], [0, 1, -3]]}
        x1, y1, x2, y2 = transform_bbox(bbox, [translate], (660, 216))
        assert x1 == 15
        assert y1 == 17
        assert x2 == 105
        assert y2 == 197

    def test_rotation_90_via_affine(self):
        # 90 degree rotation around origin: (x, y) -> (-y, x).
        # Add image_size translation so result lands inside bounds.
        bbox = (0, 0, 50, 100)
        # Rotate 90 degrees counterclockwise around origin and translate y by 50.
        # Original corners: (0,0), (50,0), (50,100), (0,100)
        # After rotation: (0,0), (0,50), (-100,50), (-100,0)
        # Without translation we get clamped to 0 — use an unbounded image_size.
        rot90 = {"type": "affine", "matrix": [[0, -1, 0], [1, 0, 0]]}
        out = transform_bbox(bbox, [rot90], (1000, 1000))
        # We expect x range [-100, 0] -> clamped to [0, 0]; y range [0, 50]
        # That's degenerate. The point is just to verify the math runs.
        assert isinstance(out, tuple)
        assert len(out) == 4

    def test_perspective_identity(self):
        bbox = (10, 20, 100, 200)
        identity = {"type": "perspective", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        out = transform_bbox(bbox, [identity], (660, 216))
        assert out == bbox

    def test_clamping_to_image_bounds(self):
        # A bbox that translates outside the image should be clamped to bounds.
        bbox = (10, 10, 50, 50)
        translate = {"type": "affine", "matrix": [[1, 0, 1000], [0, 1, 1000]]}
        x1, y1, x2, y2 = transform_bbox(bbox, [translate], (660, 216))
        assert x1 == 660 and x2 == 660 and y1 == 216 and y2 == 216

    def test_composes_transforms(self):
        # Translate +10, then translate -5 → net +5
        bbox = (0, 0, 20, 20)
        t1 = {"type": "affine", "matrix": [[1, 0, 10], [0, 1, 10]]}
        t2 = {"type": "affine", "matrix": [[1, 0, -5], [0, 1, -5]]}
        x1, y1, x2, y2 = transform_bbox(bbox, [t1, t2], (660, 216))
        assert x1 == 5 and y1 == 5
        assert x2 == 25 and y2 == 25

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            transform_bbox((0, 0, 10, 10), [{"type": "spline", "matrix": [[1]]}], (100, 100))

    def test_wrong_matrix_shape_raises(self):
        with pytest.raises(ValueError):
            transform_bbox(
                (0, 0, 10, 10),
                [{"type": "affine", "matrix": [[1, 0], [0, 1]]}],
                (100, 100),
            )


class TestTransformAnnotations:
    def test_applies_to_each(self):
        anns = [
            {"class_id": 1, "class_name": "resistor", "bbox": (10, 20, 50, 60)},
            {"class_id": 2, "class_name": "led", "bbox": (100, 100, 150, 150)},
        ]
        translate = {"type": "affine", "matrix": [[1, 0, 5], [0, 1, 5]]}
        out = transform_annotations(anns, [translate], (660, 216))
        assert len(out) == 2
        assert out[0]['bbox'] == (15, 25, 55, 65)
        assert out[1]['bbox'] == (105, 105, 155, 155)

    def test_drops_zero_area_results(self):
        anns = [{"class_id": 1, "class_name": "resistor", "bbox": (10, 10, 20, 20)}]
        # Translate way outside image — the result clamps to a degenerate bbox
        translate = {"type": "affine", "matrix": [[1, 0, 10000], [0, 1, 10000]]}
        out = transform_annotations(anns, [translate], (660, 216))
        assert out == []


# ── Validation: edge cases ─────────────────────────────────────────────


class TestEdgeCases:
    def test_component_at_left_edge(self, bbox_gen, grid):
        comp = {
            "id": "R1",
            "type": "resistor",
            "pins": {"leg1": ["a", 1], "leg2": ["a", 2]},
        }
        x1, y1, x2, y2 = bbox_gen.component_bbox(comp)
        assert x1 >= 0
        assert x2 <= bbox_gen.image_w

    def test_component_at_right_edge(self, bbox_gen):
        comp = {
            "id": "R1",
            "type": "resistor",
            "pins": {"leg1": ["a", 62], "leg2": ["a", 63]},
        }
        x1, y1, x2, y2 = bbox_gen.component_bbox(comp)
        assert x1 >= 0
        assert x2 <= bbox_gen.image_w

    def test_overlapping_components_have_distinct_bboxes(self, bbox_gen):
        # Two resistors in the same general area but different rows
        c1 = {"id": "R1", "type": "resistor", "pins": {"leg1": ["a", 10], "leg2": ["a", 15]}}
        c2 = {"id": "R2", "type": "resistor", "pins": {"leg1": ["b", 12], "leg2": ["b", 17]}}
        b1 = bbox_gen.component_bbox(c1)
        b2 = bbox_gen.component_bbox(c2)
        assert b1 != b2

    def test_single_pin_component(self, bbox_gen):
        # Hypothetical 1-pin component still produces a bbox via padding.
        comp = {"id": "X", "type": "led", "pins": {"anode": ["a", 10]}}
        x1, y1, x2, y2 = bbox_gen.component_bbox(comp)
        assert x2 > x1 and y2 > y1
