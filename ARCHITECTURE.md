# Breadboard Synthetic Data Generator V2 — Architecture Document

## Project Purpose

Generate synthetic training images of breadboard circuits for a computer vision model that classifies whether student-built circuits are correct or incorrect. The generator must be:

- **Spec-driven**: Every dimension derived from a single config, not scattered magic numbers
- **Circuit-agnostic**: Define any target circuit via JSON config; the renderer handles the rest
- **Self-validating**: Automated checks confirm output correctness without human visual inspection
- **Deterministic**: Same seed → same output, always
- **Augmentation-ready**: Clean renders feed into a realistic distortion pipeline

## V1 Failure Analysis

The first version failed because:
1. Dimensions were described in prose → AI guessed → human corrected → AI drifted
2. No validation layer — every output required manual visual inspection
3. No separation between "what to draw" (spec) and "how to draw it" (renderer)
4. Magic numbers embedded in drawing code made debugging impossible

V2 fixes all of these by separating concerns into distinct layers.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     SPEC ENGINE                          │
│  board_spec.json — single source of truth for all dims   │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                   COORDINATE SYSTEM                      │
│  grid.py — converts (row, col) → (px_x, px_y)           │
│  Owns ALL pixel math. Nothing else does pixel math.      │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                     RENDERER                             │
│  board.py    — draws base board, holes, rails, labels    │
│  components.py — draws resistors, LEDs, capacitors, ICs  │
│  wires.py    — draws wire connections between holes      │
│  Each module calls grid.py for positions. Zero pixel     │
│  math inside renderers.                                  │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                   CIRCUIT ENGINE                         │
│  circuit.py — reads circuit JSON configs                 │
│  mutations.py — generates "bad" variants from "good"     │
│  Labels are automatic: you know what you mutated.        │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                 AUGMENTATION PIPELINE                    │
│  augment.py — OpenCV post-processing                     │
│  Perspective warp, lighting jitter, blur, rotation,      │
│  shadow overlay, noise, background variation             │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                   VALIDATION LAYER                       │
│  validate.py — automated correctness checks              │
│  Asserts: hole count, spacing uniformity, rail position, │
│  component placement matches config, label correctness   │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                   BATCH GENERATOR                        │
│  generate.py — orchestrates full pipeline                │
│  Produces N images with labels, augmentation, metadata   │
│  Output: images/ + labels.json (or COCO/YOLO format)    │
└──────────────────────────────────────────────────────────┘
```

---

## Layer Details

### 1. Spec Engine (`config/board_spec.json`)

The single source of truth. Every physical dimension lives here. The spec is modeled on the **WB-102 Solderless Breadboard by Circuit Specialists** — the exact model used in the EdEon STEM Center labs.

See `config/board_spec.json` for the full spec with inline documentation. Key facts:

**WB-102 Physical Specs (from manufacturer + BB830 datasheet):**
- Overall dimensions: 165.1 x 54.0 x 8.5 mm (6.5" x 2.125" x 0.3")
- Orientation: **landscape** (wider than tall)
- Hole pitch: 2.54mm (0.1") — universal electronics standard
- Hole shape: slightly square with rounded corners (not perfectly circular)
- Terminal area: 63 columns x 10 rows, split into two halves of 5 rows (A-E top, F-J bottom)
- Center gap: 7.62mm (0.3") — sized for DIP IC packages
- Power rails: 4 total (2 top, 2 bottom), each split into two 25-hole segments with a gap near center
- Rail markings: red stripe (+), blue stripe (-)
- Rail order: red on outside, blue on inside (both top and bottom)
- Body: white/off-white ABS plastic with printed legend
- Column labels printed every 5 columns (1, 5, 10, 15, ... 60) plus 63
- 830 total tie-points: 630 terminal + 200 power rail

Key rule: `output_width_px` and `output_height_px` are COMPUTED from `real_width_mm * pixels_per_mm`. They are never set manually. The config stores real-world measurements; pixel values are derived.

**Why `pixels_per_mm` instead of DPI?** Because the renderer needs to convert mm → px constantly. Storing the conversion factor directly avoids repeated arithmetic and makes the relationship explicit. At `pixels_per_mm = 4.0`, a 2.54mm hole pitch becomes exactly 10.16px — close enough to 10px for clean rendering. Increase `pixels_per_mm` for higher-res output.

### 2. Coordinate System (`generator/grid.py`)

This is the **only module that does pixel math**. Everything else speaks in logical coordinates (row, col) and asks `grid.py` for pixel positions.

Core API:
```python
class BreadboardGrid:
    def __init__(self, spec: dict):
        """Initialize from board_spec.json"""

    def hole_center(self, row: str, col: int) -> tuple[float, float]:
        """
        Convert logical position to pixel center.
        row: 'a'-'j' for terminal, 'p1+', 'p1-', 'p2+', 'p2-' for rails
        col: 1-63
        Returns: (x_px, y_px) center of that hole
        """

    def hole_rect(self, row: str, col: int) -> tuple[float, float, float, float]:
        """Returns (x, y, w, h) bounding box of hole"""

    def board_size(self) -> tuple[int, int]:
        """Returns (width_px, height_px) of full board"""

    def all_terminal_holes(self) -> list[tuple[str, int]]:
        """Returns list of all (row, col) terminal hole positions"""

    def all_rail_holes(self) -> list[tuple[str, int]]:
        """Returns list of all (rail_id, col) power rail positions"""

    def connected_holes(self, row: str, col: int) -> list[tuple[str, int]]:
        """
        Returns all holes electrically connected to this one.
        Terminal holes: same column, same half (a-e or f-j)
        Rail holes: entire rail strip
        This is critical for mutation validation.
        """
```

**Why this matters**: In V1, pixel positions were computed inline in drawing code. When something was off by a few pixels, you couldn't tell if the bug was in the math or the drawing. By isolating all coordinate logic here, you can unit test it independently:

```python
def test_hole_spacing():
    grid = BreadboardGrid(spec)
    a1 = grid.hole_center('a', 1)
    a2 = grid.hole_center('a', 2)
    expected_pitch = spec['holes']['pitch_mm'] * spec['board']['pixels_per_mm']
    assert abs((a2[0] - a1[0]) - expected_pitch) < 0.01
```

### 3. Renderer Modules

Each renderer module takes a PIL Image and a BreadboardGrid, draws its layer, and returns the modified image. No pixel math — only calls to grid.py.

**`generator/board.py`** — Board base
- Draws the board rectangle with rounded corners
- Fills the center gap channel
- Draws column number labels (1-63) and row letter labels (a-j)
- Draws power rail color stripes

**`generator/holes.py`** — Hole grid
- Iterates `grid.all_terminal_holes()` and `grid.all_rail_holes()`
- Draws each hole as an antialiased circle at `grid.hole_center(row, col)`
- Optionally draws a subtle inner shadow for depth

**`generator/components.py`** — Electronic components
- Each component type is a class with a `draw(img, grid, placement)` method
- Supported types (extensible):
  - `Wire`: colored line between two holes, with slight droop curve
  - `Resistor`: rectangle body with color band stripes, two legs
  - `LED`: dome shape with two legs, color-tinted
  - `Capacitor`: cylindrical body or disc, two legs
  - `IC_Chip`: rectangular DIP package straddling center gap, N pins
  - `Button`: square tactile switch, 4 pins
  - `Jumper`: U-shaped wire for short connections
- Each component validates its placement against the grid (e.g., an IC must straddle the center gap)

**`generator/wires.py`** — Wire routing
- Draws wires as colored lines/curves between hole positions
- Supports: straight, L-shaped, and bezier-curved routing
- Wire colors are randomized from a realistic palette (red, black, yellow, green, blue, white, orange)
- Wires have thickness variation and slight opacity variation for realism
- Wire-over-wire overlap is handled with slight z-order offset

### 4. Circuit Engine

**`generator/circuit.py`** — Circuit definition and rendering

Reads a circuit config JSON and orchestrates rendering:

```json
{
  "name": "simple_led_circuit",
  "description": "LED with current-limiting resistor powered from rail",
  "components": [
    {
      "id": "R1",
      "type": "resistor",
      "value": "220",
      "pins": { "leg1": ["a", 10], "leg2": ["a", 15] }
    },
    {
      "id": "LED1",
      "type": "led",
      "color": "red",
      "pins": { "anode": ["a", 20], "cathode": ["b", 20] }
    }
  ],
  "wires": [
    { "from": ["p1+", 10], "to": ["a", 10], "color": "red" },
    { "from": ["a", 15], "to": ["a", 20], "color": "yellow" },
    { "from": ["b", 20], "to": ["p1-", 20], "color": "black" }
  ]
}
```

**`generator/mutations.py`** — Error injection

Takes a valid circuit config and produces mutated (incorrect) versions:

```python
class MutationEngine:
    def remove_component(self, circuit, component_id=None) -> tuple[dict, dict]:
        """Remove a random (or specified) component. Returns (mutated_circuit, mutation_record)"""

    def wrong_position(self, circuit, component_id=None, max_shift=5) -> tuple[dict, dict]:
        """Shift a component to a nearby wrong position"""

    def wrong_connection(self, circuit, wire_index=None) -> tuple[dict, dict]:
        """Rewire a connection to a wrong hole"""

    def swap_polarity(self, circuit) -> tuple[dict, dict]:
        """Swap +/- rail connections (common student mistake)"""

    def extra_component(self, circuit) -> tuple[dict, dict]:
        """Add a spurious component (another common mistake)"""

    def compound_mutation(self, circuit, n_mutations=2) -> tuple[dict, dict]:
        """Apply multiple mutations for harder examples"""
```

Every mutation function returns a `mutation_record` dict describing exactly what changed — this becomes part of the label metadata.

### 5. Augmentation Pipeline (`generator/augment.py`)

Post-processing to close the domain gap between synthetic and real photos. All augmentations are parameterized with ranges for randomization.

```python
class AugmentationPipeline:
    def __init__(self, config: dict):
        """Config specifies ranges for each augmentation"""

    def perspective_warp(self, img, max_angle_deg=15) -> np.ndarray:
        """Simulate camera not being perfectly overhead"""

    def lighting_jitter(self, img, brightness_range=(-30, 30), contrast_range=(0.8, 1.2)):
        """Random brightness and contrast shifts"""

    def gaussian_blur(self, img, kernel_range=(1, 5)):
        """Simulate slight out-of-focus"""

    def rotation(self, img, max_angle_deg=5):
        """Slight in-plane rotation"""

    def add_shadow(self, img, shadow_intensity_range=(0.3, 0.7)):
        """Overlay a random gradient shadow"""

    def add_noise(self, img, noise_std_range=(5, 20)):
        """Gaussian sensor noise"""

    def background_variation(self, img, bg_colors=None):
        """Place board on different colored surfaces"""

    def apply_random(self, img, n_augmentations=(2, 4)) -> tuple[np.ndarray, dict]:
        """Apply a random subset of augmentations. Returns (image, augmentation_record)"""
```

### 6. Validation Layer (`generator/validate.py`)

Automated checks that run after rendering. These replace human visual inspection.

```python
class BoardValidator:
    def __init__(self, spec: dict, grid: BreadboardGrid):
        pass

    def validate_board(self, img: np.ndarray) -> list[str]:
        """
        Returns list of validation errors (empty = pass).
        Checks:
        - Image dimensions match spec
        - Hole count matches expected (sample check via template matching)
        - Holes are on the expected pixel grid (within tolerance)
        - Power rail stripes are in expected positions (color sampling)
        - Center gap is clear (no unexpected content)
        """

    def validate_circuit(self, img: np.ndarray, circuit_config: dict) -> list[str]:
        """
        Checks:
        - Each component in config has visible pixels near its expected position
        - Each wire in config has colored pixels along its expected path
        - No unexpected large colored regions (rendering artifacts)
        """

    def validate_mutation(self, original_config: dict, mutated_config: dict, mutation_record: dict) -> list[str]:
        """
        Checks:
        - Mutation record is consistent with config diff
        - Mutated circuit actually differs from original
        - Mutation is physically possible (component still fits on board)
        """
```

### 7. Batch Generator (`generate.py`)

The main entry point that orchestrates everything:

```python
def generate_dataset(
    circuit_config_path: str,
    board_spec_path: str,
    output_dir: str,
    n_correct: int = 1000,
    n_incorrect: int = 4000,
    augmentations_per_image: int = 3,
    seed: int = 42,
    validate: bool = True
):
    """
    Generates a full dataset:
    1. Load spec and circuit config
    2. For each correct image:
       a. Render clean board + circuit
       b. Apply N random augmentations → N output images
       c. Validate each
    3. For each incorrect image:
       a. Pick a random mutation type
       b. Mutate the circuit config
       c. Render the mutated circuit
       d. Apply N random augmentations
       e. Validate each
    4. Save images + labels.json with full metadata
    """
```

Output structure:
```
output/
├── images/
│   ├── correct_0001_aug0.png
│   ├── correct_0001_aug1.png
│   ├── incorrect_0001_aug0.png
│   └── ...
├── labels.json          # master label file
├── metadata.json        # generation parameters, seed, timestamp
└── validation_report.txt # any validation warnings
```

`labels.json` format:
```json
[
  {
    "filename": "correct_0001_aug0.png",
    "label": "correct",
    "circuit_config": "simple_led_circuit",
    "mutations": [],
    "augmentations": ["perspective_warp", "lighting_jitter", "gaussian_blur"],
    "augmentation_params": { ... },
    "seed": 42
  },
  {
    "filename": "incorrect_0023_aug1.png",
    "label": "incorrect",
    "circuit_config": "simple_led_circuit",
    "mutations": [
      { "type": "wrong_connection", "wire_index": 2, "original": ["b", 20], "mutated": ["c", 22] }
    ],
    "augmentations": ["rotation", "add_noise"],
    "augmentation_params": { ... },
    "seed": 42
  }
]
```

---

## Directory Structure

```
breadboard_generator/
├── CLAUDE.md                    # Claude Code context (see separate file)
├── ARCHITECTURE.md              # This document
├── requirements.txt             # pillow, opencv-python, numpy, matplotlib
├── reference/
│   └── wb102_photo.jpg          # Real photo of the WB-102 (visual reference only)
├── config/
│   ├── board_spec.json          # Board dimensions (single source of truth)
│   ├── augmentation_config.json # Augmentation parameter ranges
│   └── circuits/
│       ├── simple_led.json      # Example circuit configs
│       └── resistor_divider.json
├── generator/
│   ├── __init__.py
│   ├── grid.py                  # Coordinate system (ONLY pixel math lives here)
│   ├── board.py                 # Board base renderer
│   ├── holes.py                 # Hole grid renderer
│   ├── components.py            # Component renderers (Wire, Resistor, LED, etc.)
│   ├── wires.py                 # Wire routing and drawing
│   ├── circuit.py               # Circuit config loader + orchestrator
│   ├── mutations.py             # Error injection engine
│   ├── augment.py               # OpenCV augmentation pipeline
│   └── validate.py              # Automated output validation
├── generate.py                  # Main entry point / batch generator
├── tests/
│   ├── test_grid.py             # Unit tests for coordinate system
│   ├── test_board.py            # Tests for board rendering
│   ├── test_mutations.py        # Tests for mutation correctness
│   └── test_validate.py         # Tests for validation layer
└── output/                      # Generated datasets land here
    └── .gitkeep
```

---

## Implementation Order

Build and test each layer before moving to the next. Each phase has a concrete, verifiable deliverable.

### Phase 1: Spec Engine + Coordinate System + Blank Board
- Create `board_spec.json` with all measurements
- Implement `grid.py` with full coordinate API
- Implement `board.py` + `holes.py` to render blank board
- Write `test_grid.py` — assert hole positions, spacing, board dimensions
- Write `validate.py` basics — assert hole count and spacing on rendered image
- **Deliverable**: One PNG of a blank breadboard that passes all validation checks

### Phase 2: Component Rendering
- Implement `components.py` with Wire, Resistor, LED classes
- Implement `wires.py` with basic straight + L-shaped routing
- **Deliverable**: Blank board + manually placed components that look realistic

### Phase 3: Circuit Config System
- Implement `circuit.py` to load JSON configs and orchestrate rendering
- Create 2-3 example circuit configs
- **Deliverable**: Change the JSON, get a different circuit image. No code changes needed.

### Phase 4: Mutation Engine
- Implement `mutations.py` with all mutation types
- Write `test_mutations.py` — assert mutations produce valid-but-different configs
- **Deliverable**: Given a correct circuit, automatically generate 5+ distinct error variants with labels

### Phase 5: Augmentation Pipeline
- Implement `augment.py` with all augmentation types
- Create `augmentation_config.json` with tunable ranges
- **Deliverable**: Same circuit rendered with 10 different realistic-looking augmentations

### Phase 6: Batch Generation + Full Validation
- Implement `generate.py` orchestrator
- Full validation pipeline on every generated image
- **Deliverable**: Generate 5,000+ labeled images in one command, with validation report

---

## Design Principles

1. **No magic numbers**: Every dimension traces back to `board_spec.json`
2. **No pixel math outside grid.py**: If you're computing pixels, it goes in grid.py
3. **Validation over visual inspection**: If you can't write a test for it, redesign it
4. **Seed everything**: numpy and random both seeded for reproducibility
5. **Fail loud**: Validation errors raise exceptions, not warnings
6. **Config over code**: Changing what circuit to render should never require changing Python code
