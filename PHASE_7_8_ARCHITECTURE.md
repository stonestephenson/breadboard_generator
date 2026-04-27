# PHASE 7-8 ARCHITECTURE — Bounding Boxes + CycleGAN Style Transfer

## Context

Phases 1-6 are complete. The synthetic breadboard image generator produces labeled datasets
with 6 mutation types and 7 augmentation types. Two new requirements:

1. Add bounding box annotations over circuit components
2. Make synthetic images look photorealistic via CycleGAN style transfer

---

## Phase 7: Bounding Box Annotations

### Goal

Every generated image includes bounding box annotations for each visible component,
exported in both COCO JSON and YOLO txt format. The user selects the format via a CLI flag.

### Why This Is Easy

The circuit config already stores pin positions for every component. The BreadboardGrid
already converts those to pixel coordinates. Bounding boxes are just the rectangular
extent of each component's pixel footprint plus some padding.

### Object Classes

| Class ID | Class Name  | Description                                    |
|----------|-------------|------------------------------------------------|
| 0        | breadboard  | The entire board — always present, full extent |
| 1        | resistor    | Resistor body + leads                          |
| 2        | led         | LED dome + leads                               |
| 3        | wire        | Wire between two holes                         |
| 4        | capacitor   | Capacitor body + leads (when implemented)      |
| 5        | ic_chip     | IC package (when implemented)                  |

### Implementation

**`generator/annotations.py`** — New module. Core API:

```python
class BoundingBoxGenerator:
    def __init__(self, grid: BreadboardGrid, spec: dict):
        pass

    def component_bbox(self, component: dict) -> tuple[int, int, int, int]:
        """
        Given a component dict from the circuit config, return (x_min, y_min, x_max, y_max)
        in pixel coordinates. Adds padding based on component type to account for
        visual extent beyond pin holes (LED dome, resistor body, etc.)
        """

    def wire_bbox(self, wire: dict) -> tuple[int, int, int, int]:
        """
        Given a wire dict, return bounding box covering the full wire path.
        For L-shaped wires, the bbox covers both segments.
        """

    def board_bbox(self) -> tuple[int, int, int, int]:
        """Returns bounding box for the entire breadboard."""

    def generate_annotations(self, circuit_config: dict) -> list[dict]:
        """
        Returns list of annotation dicts:
        [
            {"class_id": 0, "class_name": "breadboard", "bbox": (x1, y1, x2, y2)},
            {"class_id": 1, "class_name": "resistor", "bbox": (x1, y1, x2, y2), "component_id": "R1"},
            ...
        ]
        """

    def to_coco(self, annotations: list[dict], image_id: int, image_size: tuple) -> dict:
        """Convert to COCO JSON format (x, y, width, height)."""

    def to_yolo(self, annotations: list[dict], image_size: tuple) -> str:
        """Convert to YOLO txt format (class_id, x_center, y_center, width, height — all normalized 0-1)."""
```

### Padding per Component Type

Components are visually larger than their pin holes. Padding values (in mm, converted to px via spec):

| Component  | Padding X (mm) | Padding Y (mm) | Notes                           |
|------------|-----------------|-----------------|----------------------------------|
| resistor   | 1.0             | 1.5             | Body extends beyond pin holes    |
| led        | 2.0             | 2.0             | Dome is wider than pin spacing   |
| wire       | 0.5             | 0.5             | Minimal — just the wire itself   |
| capacitor  | 1.0             | 1.5             | Similar to resistor              |
| breadboard | 0.0             | 0.0             | Full board extent, no padding    |

### Integration with generate.py

Extend the batch generator to:
1. Compute bounding boxes for each image after rendering (pre-augmentation)
2. If augmentations include perspective warp or rotation, transform the bounding boxes accordingly
3. Export annotations in the user-selected format alongside images
4. Add `--annotation-format` flag: `coco`, `yolo`, or `both`

### Output Structure (extended)

```
output/dataset_v1/
├── images/
│   ├── correct_0001_aug0.png
│   └── ...
├── labels/                          # NEW — YOLO format
│   ├── correct_0001_aug0.txt
│   └── ...
├── annotations.json                 # NEW — COCO format (single file for all images)
├── labels.json                      # Existing mutation/classification labels
├── metadata.json
└── validation_report.txt
```

### Bounding Box Validation

Add to `generator/validate.py`:
- Every component in the circuit config has a corresponding bounding box
- No bounding box extends outside image bounds
- No bounding box has zero or negative area
- Bounding boxes for components at known positions are within tolerance of expected pixel coords

### Tests

`tests/test_annotations.py`:
- Bounding box computation for each component type
- COCO format output validity
- YOLO format output validity (all values 0-1, correct class IDs)
- Bounding box transformation under augmentation
- Edge cases: components near board edges, overlapping components

---

## Phase 8: CycleGAN Style Transfer

### Goal

Train a CycleGAN to translate synthetic breadboard images into photorealistic ones,
preserving component positions and spatial structure so that bounding box annotations
remain valid.

### How CycleGAN Works (Brief)

CycleGAN learns two mapping functions:
- G: Synthetic → Realistic (this is what we want)
- F: Realistic → Synthetic (learned simultaneously to enforce consistency)

It trains on UNPAIRED images from both domains. This means:
- Domain A: Synthetic rendered images (from our pipeline)
- Domain B: Real breadboard photos (from your camera)
- No need to match specific synthetic images to specific real photos

The cycle consistency loss ensures that G(synthetic) looks realistic while preserving
the structural content — components stay where they are, wiring layout is maintained.

### Data Collection Plan

**Synthetic images (Domain A):**
Generate 500-1000 images using the existing pipeline. Include:
- Various circuits (all 3 example configs + mutations)
- Light augmentation only (slight rotation, minor lighting) — don't over-augment since
  the CycleGAN will learn the realistic appearance
- Both correct and incorrect circuits (variety of component layouts)

**Real photos (Domain B):**
Collect 100-200 photos of real WB-102 breadboards. Guidelines:
- Mix of empty boards and boards with circuits
- Multiple lighting conditions (fluorescent, natural, desk lamp, mixed)
- Multiple angles (overhead, slight tilt 5-15°, various rotations)
- Multiple backgrounds (dark desk, light desk, lab bench, textured surface)
- Various wire colors and component configurations
- Include some messy/realistic student-quality builds
- Ensure the full breadboard is visible in every shot
- Mix of phone cameras (this is what students will use)
- Resolution: whatever the phone shoots, will be resized during training

Photo naming convention: `real_001.jpg` through `real_200.jpg` in `data/real/`

### Project Structure (new additions)

```
breadboard_generator/
├── ... (existing project files)
├── data/
│   ├── synthetic/          # Generated by pipeline for CycleGAN training
│   │   ├── train/          # 80% of synthetic images
│   │   └── test/           # 20% held out
│   ├── real/               # Real photos you collect
│   │   ├── train/          # 80% of real photos
│   │   └── test/           # 20% held out
│   └── prepare_data.py     # Script to generate synthetic set + split both domains
├── cyclegan/
│   ├── README.md           # CycleGAN-specific setup and training instructions
│   ├── train.py            # Training script (wraps pytorch-CycleGAN-and-pix2pix)
│   ├── test.py             # Generate translated images from synthetic inputs
│   ├── config.py           # Training hyperparameters
│   └── evaluate.py         # Structural similarity metrics (SSIM, component position drift)
└── scripts/
    └── apply_stylization.py  # Post-training: apply learned G to full dataset
```

### Implementation Plan

**Phase 8a — Setup and Data Preparation**
1. Install pytorch-CycleGAN-and-pix2pix (well-maintained open source implementation)
2. Create `data/prepare_data.py` to generate synthetic training set from pipeline
3. Create folder structure for real photos
4. Write a preprocessing script to resize and normalize both domains to same resolution
5. Split both domains into train/test (80/20)

**Phase 8b — Data Collection**
1. Photograph 100-200 real breadboard images following the guidelines above
2. Place in `data/real/`
3. Run preprocessing to normalize sizes

**Phase 8c — CycleGAN Training**
1. Configure training hyperparameters in `config.py`:
   - Image size: 256x256 or 512x512 (balance quality vs training time)
   - Batch size: 1 (standard for CycleGAN)
   - Learning rate: 0.0002 (standard)
   - Epochs: 100-200 (monitor for convergence)
   - Lambda cycle: 10.0 (cycle consistency weight)
   - Lambda identity: 0.5 (helps preserve color)
2. Train on GPU (required — CycleGAN is slow on CPU)
3. Monitor training with sample outputs every N epochs
4. Save checkpoints every 10 epochs

**Phase 8d — Validation**
Critical: verify the style transfer preserves spatial structure.
1. Generate translated images from synthetic test set
2. Run bounding box predictions on translated images using the annotations from the synthetic source
3. Compute structural similarity (SSIM) between synthetic and translated images
4. Manually inspect 20-30 samples: are components still where they should be?
5. Measure bounding box drift: overlay synthetic annotations on translated images and check alignment
6. If drift > N pixels, increase cycle consistency weight and retrain

**Phase 8e — Pipeline Integration**
1. Create `scripts/apply_stylization.py`:
   - Takes a synthetic dataset (from generate.py output)
   - Applies the trained generator G to every image
   - Copies labels.json and bounding box annotations unchanged (they should still be valid)
   - Outputs a new "stylized" dataset directory
2. Optionally integrate as a flag in generate.py: `--stylize` applies the trained model as a final step

### Hardware Requirements

CycleGAN training needs a GPU. Options:
- Google Colab (free tier has GPU, Pro is $10/month for better GPUs)
- University GPU cluster (check if SSU has one)
- Your own GPU if you have one

Training time estimate: 4-12 hours on a decent GPU (RTX 3060 or better) for 200 epochs.

### Risk Mitigation

**Risk: CycleGAN distorts component positions**
Mitigation: High cycle consistency weight (lambda=10), identity loss, structural validation in Phase 8d.
If positions drift too much, consider pix2pix instead (requires paired data but preserves structure better).

**Risk: Not enough real photos**
Mitigation: Start with 100, evaluate quality, collect more if needed. CycleGAN has been shown to
work with as few as 50 images per domain, though quality improves with more.

**Risk: Training instability**
Mitigation: Use the well-tested pytorch-CycleGAN-and-pix2pix codebase rather than implementing from scratch.
Monitor discriminator/generator loss balance. Use learning rate scheduling.

**Risk: Generated images have CycleGAN artifacts (checkerboard patterns, color bleeding)**
Mitigation: Use ResNet generator (not U-Net) for higher quality. Increase training time.
Apply light post-processing (slight Gaussian blur) to smooth artifacts.

---

## Implementation Order

1. **Phase 7** first — bounding boxes are independent of style transfer and immediately useful
2. **Phase 8a** — set up CycleGAN infrastructure and data prep scripts
3. **Phase 8b** — collect real photos (do this in parallel with Phase 8a)
4. **Phase 8c** — train the CycleGAN
5. **Phase 8d** — validate spatial preservation
6. **Phase 8e** — integrate into pipeline

Phase 7 is probably 1-2 Claude Code sessions.
Phase 8 spans multiple days due to data collection and GPU training time.

---

## Updated CLAUDE.md Additions

Add these to CLAUDE.md:

```
## Phase 7: Bounding Box Annotations

generator/annotations.py generates bounding boxes for all components in a circuit config.
Supports COCO JSON and YOLO txt export formats. Bounding boxes are computed from pin
positions via grid.py with per-component-type padding for visual extent.
CLI flag: --annotation-format coco|yolo|both

## Phase 8: CycleGAN Style Transfer

Post-processing step that translates synthetic renders to photorealistic images.
Uses pytorch-CycleGAN-and-pix2pix. Domain A = synthetic images, Domain B = real photos.
The trained generator is applied after rendering, before final export. Bounding box
annotations carry through unchanged because CycleGAN preserves spatial structure.
```
