"""
Bounding box annotation generator — produces COCO and YOLO format
annotations for synthetic breadboard images.

Computes axis-aligned bounding boxes for the board, each component, and
each wire from pin positions retrieved via grid.py, plus per-component-type
padding (in mm, converted to pixels via the spec). All pixel math is
delegated to grid.py — this module only does padding arithmetic and
format conversion.
"""

from typing import Optional

import numpy as np

from generator.grid import BreadboardGrid


CLASS_MAP: dict[str, int] = {
    "breadboard": 0,
    "resistor": 1,
    "led": 2,
    "wire": 3,
    "capacitor": 4,
    "ic_chip": 5,
}

CLASS_NAMES: dict[int, str] = {v: k for k, v in CLASS_MAP.items()}

# Padding in millimeters per axis. Components are visually larger than
# their pin holes — these values cover bodies, domes, and lead radius.
PADDING_MM: dict[str, dict[str, float]] = {
    "resistor":   {"x": 1.0, "y": 1.5},
    "led":        {"x": 2.0, "y": 2.0},
    "wire":       {"x": 0.5, "y": 0.5},
    "capacitor":  {"x": 1.0, "y": 1.5},
    "ic_chip":    {"x": 1.0, "y": 1.5},
    "breadboard": {"x": 0.0, "y": 0.0},
}


class BoundingBoxGenerator:
    """Computes bounding boxes for circuit components and exports them as
    COCO JSON or YOLO txt records."""

    def __init__(self, grid: BreadboardGrid, spec: dict):
        """
        Args:
            grid: BreadboardGrid (provides hole_center and board_size).
            spec: Parsed board_spec.json dict.
        """
        self.grid = grid
        self.spec = spec
        self.ppmm = grid.ppmm
        self.image_w = grid.board_width_px
        self.image_h = grid.board_height_px

    # ── Helpers ────────────────────────────────────────────────────────

    def _padding_px(self, comp_type: str) -> tuple[float, float]:
        """Return (pad_x_px, pad_y_px) for a component type."""
        pad = PADDING_MM.get(comp_type, {"x": 1.0, "y": 1.5})
        return pad["x"] * self.ppmm, pad["y"] * self.ppmm

    def _pin_positions(self, component: dict) -> list[tuple[float, float]]:
        """Return pixel centers of every pin in a component dict."""
        pins = component.get("pins", {})
        positions: list[tuple[float, float]] = []
        for pin_val in pins.values():
            if isinstance(pin_val, (list, tuple)) and len(pin_val) == 2:
                positions.append(self.grid.hole_center(pin_val[0], pin_val[1]))
        return positions

    def _clamp_to_image(
        self, x_min: float, y_min: float, x_max: float, y_max: float,
        image_size: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int, int, int]:
        """Round to int and clamp to image bounds. Returns (x_min,y_min,x_max,y_max)."""
        w, h = image_size if image_size is not None else (self.image_w, self.image_h)
        x_min_i = max(0, min(w, int(round(x_min))))
        y_min_i = max(0, min(h, int(round(y_min))))
        x_max_i = max(0, min(w, int(round(x_max))))
        y_max_i = max(0, min(h, int(round(y_max))))
        # Guarantee ordering
        if x_max_i < x_min_i:
            x_min_i, x_max_i = x_max_i, x_min_i
        if y_max_i < y_min_i:
            y_min_i, y_max_i = y_max_i, y_min_i
        return (x_min_i, y_min_i, x_max_i, y_max_i)

    # ── Per-element bounding boxes ─────────────────────────────────────

    def component_bbox(self, component: dict) -> tuple[int, int, int, int]:
        """
        Compute (x_min, y_min, x_max, y_max) in pixels for a component.

        Bbox is the axis-aligned extent of all pin centers, expanded by the
        component-type padding. Clamped to image bounds.
        """
        ctype = component.get("type", "")
        positions = self._pin_positions(component)
        if not positions:
            raise ValueError(
                f"Component {component.get('id', '?')} has no resolvable pins"
            )
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        pad_x, pad_y = self._padding_px(ctype)
        return self._clamp_to_image(
            min(xs) - pad_x,
            min(ys) - pad_y,
            max(xs) + pad_x,
            max(ys) + pad_y,
        )

    def wire_bbox(self, wire: dict) -> tuple[int, int, int, int]:
        """
        Compute (x_min, y_min, x_max, y_max) in pixels for a wire.

        Covers both endpoints (and therefore both segments of an L-shaped wire,
        whose elbow is at one of the endpoint columns/rows). Clamped to image bounds.
        """
        start = wire["from"]
        end = wire["to"]
        sx, sy = self.grid.hole_center(start[0], start[1])
        ex, ey = self.grid.hole_center(end[0], end[1])
        pad_x, pad_y = self._padding_px("wire")
        return self._clamp_to_image(
            min(sx, ex) - pad_x,
            min(sy, ey) - pad_y,
            max(sx, ex) + pad_x,
            max(sy, ey) + pad_y,
        )

    def board_bbox(self) -> tuple[int, int, int, int]:
        """Bounding box for the entire breadboard (full image extent)."""
        return (0, 0, int(self.image_w), int(self.image_h))

    # ── Annotation generation ──────────────────────────────────────────

    def generate_annotations(self, circuit_config: dict) -> list[dict]:
        """
        Compute annotations for every visible element in a circuit config.

        Returns:
            List of dicts. Always includes the board itself first, then every
            component, then every wire. Each dict has at minimum:
                {"class_id": int, "class_name": str, "bbox": (x1, y1, x2, y2)}
            Components additionally carry "component_id"; wires carry "wire_index".
        """
        annotations: list[dict] = [{
            "class_id": CLASS_MAP["breadboard"],
            "class_name": "breadboard",
            "bbox": self.board_bbox(),
        }]

        for comp in circuit_config.get("components", []):
            ctype = comp.get("type", "")
            class_id = CLASS_MAP.get(ctype)
            if class_id is None:
                # Unknown component type — skip rather than crash so that
                # forward-compat configs don't break the pipeline.
                continue
            annotations.append({
                "class_id": class_id,
                "class_name": ctype,
                "bbox": self.component_bbox(comp),
                "component_id": comp.get("id"),
            })

        for i, wire in enumerate(circuit_config.get("wires", [])):
            annotations.append({
                "class_id": CLASS_MAP["wire"],
                "class_name": "wire",
                "bbox": self.wire_bbox(wire),
                "wire_index": i,
            })

        return annotations

    # ── Format converters ──────────────────────────────────────────────

    def to_coco(
        self,
        annotations: list[dict],
        image_id: int,
        image_size: tuple[int, int],
    ) -> dict:
        """
        Convert annotations to a COCO per-image record.

        COCO bbox format is [x, y, width, height] (top-left + size).

        Returns:
            {"image": {...}, "annotations": [...]} — caller should aggregate
            multiple per-image records via coco_dataset() to build a full file.
        """
        w, h = image_size
        coco_annots: list[dict] = []
        for i, a in enumerate(annotations):
            x1, y1, x2, y2 = a["bbox"]
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            entry: dict = {
                "id": i,
                "image_id": image_id,
                "category_id": int(a["class_id"]),
                "bbox": [int(x1), int(y1), int(bw), int(bh)],
                "area": int(bw * bh),
                "iscrowd": 0,
            }
            if "component_id" in a and a["component_id"] is not None:
                entry["component_id"] = a["component_id"]
            if "wire_index" in a:
                entry["wire_index"] = a["wire_index"]
            coco_annots.append(entry)
        return {
            "image": {
                "id": int(image_id),
                "width": int(w),
                "height": int(h),
            },
            "annotations": coco_annots,
        }

    def to_yolo(self, annotations: list[dict], image_size: tuple[int, int]) -> str:
        """
        Convert annotations to a YOLO-format text block.

        One line per annotation: ``class_id x_center y_center width height``,
        all spatial values normalized to [0, 1]. Trailing newline included only
        if there is at least one annotation (caller can append as needed).
        """
        w, h = image_size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size: {image_size}")

        lines: list[str] = []
        for a in annotations:
            x1, y1, x2, y2 = a["bbox"]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            nw = bw / w
            nh = bh / h
            lines.append(
                f"{int(a['class_id'])} "
                f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            )
        return "\n".join(lines)


# ── Geometric transforms (for augmentation propagation) ──────────────────


def transform_bbox(
    bbox: tuple[int, int, int, int],
    transforms: list[dict],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Apply a sequence of geometric transforms to a bbox and return the
    axis-aligned bbox of the transformed corners.

    Args:
        bbox: (x_min, y_min, x_max, y_max) in source pixel coordinates.
        transforms: List of dicts, each with 'type' and 'matrix' fields.
            type == 'perspective' → 3x3 homography (matrix as 3x3 nested list).
            type == 'affine'      → 2x3 affine matrix (matrix as 2x3 nested list).
        image_size: (w, h) of the destination image. Result is clamped to it.

    Returns:
        Axis-aligned (x_min, y_min, x_max, y_max), integer-rounded and clamped.
    """
    x1, y1, x2, y2 = bbox
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float64,
    )

    for t in transforms:
        ttype = t["type"]
        M = np.array(t["matrix"], dtype=np.float64)
        homog = np.hstack([corners, np.ones((corners.shape[0], 1))])

        if ttype == "perspective":
            if M.shape != (3, 3):
                raise ValueError(f"Perspective matrix must be 3x3, got {M.shape}")
            warped = (M @ homog.T).T
            # Avoid divide-by-zero — if any z is 0, leave the corner in place.
            zs = warped[:, 2:3]
            zs_safe = np.where(np.abs(zs) < 1e-9, 1.0, zs)
            corners = warped[:, :2] / zs_safe
        elif ttype == "affine":
            if M.shape != (2, 3):
                raise ValueError(f"Affine matrix must be 2x3, got {M.shape}")
            corners = (M @ homog.T).T
        else:
            raise ValueError(f"Unknown transform type: {ttype!r}")

    xs = corners[:, 0]
    ys = corners[:, 1]
    w, h = image_size
    x_min = max(0, min(w, int(round(float(np.min(xs))))))
    y_min = max(0, min(h, int(round(float(np.min(ys))))))
    x_max = max(0, min(w, int(round(float(np.max(xs))))))
    y_max = max(0, min(h, int(round(float(np.max(ys))))))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return (x_min, y_min, x_max, y_max)


def transform_annotations(
    annotations: list[dict],
    transforms: list[dict],
    image_size: tuple[int, int],
) -> list[dict]:
    """Apply transforms to every annotation's bbox. Drops zero-area results."""
    out: list[dict] = []
    for a in annotations:
        new_bbox = transform_bbox(a["bbox"], transforms, image_size)
        x1, y1, x2, y2 = new_bbox
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        new_a = dict(a)
        new_a["bbox"] = new_bbox
        out.append(new_a)
    return out


# ── COCO dataset aggregation ────────────────────────────────────────────


def coco_dataset(per_image_records: list[dict], description: str = "") -> dict:
    """
    Build a full COCO dataset JSON from per-image records returned by
    BoundingBoxGenerator.to_coco().

    Annotation IDs are renumbered globally so they are unique across the dataset.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 0
    for record in per_image_records:
        images.append(dict(record["image"]))
        for a in record["annotations"]:
            entry = dict(a)
            entry["id"] = ann_id
            ann_id += 1
            annotations.append(entry)
    categories = [
        {"id": cid, "name": name, "supercategory": "circuit"}
        for name, cid in CLASS_MAP.items()
    ]
    return {
        "info": {
            "description": description or "Synthetic breadboard circuit dataset",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
