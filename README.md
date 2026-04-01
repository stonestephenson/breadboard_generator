# Breadboard Synthetic Data Generator

Generate labeled synthetic training images of breadboard circuits for a computer vision classifier that detects correct vs. incorrect student-built circuits.

## Overview

This project programmatically renders realistic breadboard images and automatically labels them for supervised learning. Define a target circuit in JSON, and the pipeline generates thousands of labeled images: correct builds with random visual augmentations, and incorrect builds with injected wiring/component errors.

**Key design principles:**

- **Spec-driven** — Every physical dimension comes from a single config file (`board_spec.json`). No magic numbers in rendering code.
- **Config-driven** — Change what circuit to render by editing a JSON file, not Python code. Same for augmentation intensity, board resolution, and component colors.
- **Self-validating** — Automated checks run on every generated image. No manual visual inspection required.
- **Reproducible** — All randomness is seeded. Same seed + same config = identical output, always.

## Features

- **Blank board rendering** — Accurate WB-102 breadboard with holes, power rails, stripes, labels, and center DIP channel
- **Component placement** — Resistors with color bands, LEDs with dome highlights, both with wire leads
- **Wire routing** — Straight and L-shaped wire paths in 7 colors
- **Circuit configs** — Define any circuit as JSON; the renderer handles placement, wiring, and validation
- **6 mutation types** — Remove component, wrong position, wrong connection, swapped polarity, extra component, compound mutations
- **7 augmentation types** — Perspective warp, lighting jitter, Gaussian blur, rotation, shadows, sensor noise, background variation
- **Batch generation** — One command produces thousands of labeled images with full metadata
- **Automatic labeling** — Every mutation is tracked, so labels are generated as a byproduct of the pipeline

## Quick Start

**Prerequisites:** Python 3.10+, pip

```bash
# Clone and setup
git clone <repo-url>
cd breadboard_generator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate a dataset
python generate.py \
  --circuit config/circuits/simple_led.json \
  --n-correct 50 --n-incorrect 200 \
  --seed 42 --output output/my_dataset/
```

Output lands in `output/my_dataset/` with `images/`, `labels.json`, `metadata.json`, and `validation_report.txt`.

## Usage

### Generate a full training dataset

```bash
python generate.py \
  --circuit config/circuits/simple_led.json \
  --n-correct 1000 \
  --n-incorrect 4000 \
  --seed 42 \
  --output output/dataset_v1/
```

### Generate from a different circuit

```bash
python generate.py \
  --circuit config/circuits/dual_led.json \
  --n-correct 500 --n-incorrect 2000 \
  --output output/dual_led_dataset/
```

### Create a custom circuit config

Create a JSON file in `config/circuits/`. Here is a complete example with all component types and fields:

```json
{
  "name": "my_custom_circuit",
  "description": "Red LED with 220-ohm resistor powered from top rail",
  "components": [
    {
      "id": "R1",
      "type": "resistor",
      "value": "220",
      "pins": { "leg1": ["a", 10], "leg2": ["a", 15] },
      "bands": [[200, 40, 40], [200, 40, 40], [139, 69, 19], [212, 175, 55]]
    },
    {
      "id": "LED1",
      "type": "led",
      "color": "red",
      "pins": { "anode": ["d", 25], "cathode": ["e", 25] }
    }
  ],
  "wires": [
    { "from": ["p1+", 9], "to": ["a", 10], "color": "red", "routing": "L" },
    { "from": ["a", 15], "to": ["a", 25], "color": "yellow" },
    { "from": ["e", 25], "to": ["p1-", 26], "color": "black", "routing": "L" }
  ]
}
```

Then generate: `python generate.py --circuit config/circuits/my_custom_circuit.json ...`

### Adjust augmentation settings

Edit `config/augmentation_config.json` to tune intensity ranges, enable/disable specific augmentations, or change how many are applied per image.

### Render a single blank board

```bash
python -c "
from generator.board import render_blank_board
render_blank_board('config/board_spec.json', 'output/blank.png')
"
```

### Render a single circuit (no augmentation)

```bash
python -c "
from generator.circuit import render_circuit_to_file
render_circuit_to_file(
    'config/circuits/simple_led.json',
    'config/board_spec.json',
    'output/preview.png',
)
"
```

### Skip validation for faster generation

```bash
python generate.py --circuit config/circuits/simple_led.json \
  --n-correct 1000 --n-incorrect 4000 --no-validate \
  --output output/fast_dataset/
```

## Project Structure

```
breadboard_generator/
├── generate.py                      # Main entry point — batch dataset generation
├── requirements.txt                 # Python dependencies
├── ARCHITECTURE.md                  # Full system design and layer details
├── CLAUDE.md                        # AI assistant context and design rules
├── config/
│   ├── board_spec.json              # Board dimensions (single source of truth)
│   ├── augmentation_config.json     # Augmentation parameter ranges
│   └── circuits/
│       ├── simple_led.json          # LED + resistor circuit
│       ├── resistor_divider.json    # Two-resistor voltage divider
│       └── dual_led.json           # Two LEDs with shared resistor
├── generator/
│   ├── __init__.py
│   ├── grid.py                      # Coordinate system — ONLY module doing pixel math
│   ├── board.py                     # Board base renderer (body, stripes, labels, gap)
│   ├── holes.py                     # Hole grid renderer (terminal + rail holes)
│   ├── components.py                # Component renderers (Resistor, LED)
│   ├── wires.py                     # Wire routing and drawing (straight, L-shaped)
│   ├── circuit.py                   # Circuit config loader and render orchestrator
│   ├── mutations.py                 # Error injection engine (6 mutation types)
│   ├── augment.py                   # OpenCV augmentation pipeline (7 types)
│   └── validate.py                  # Automated correctness checks
├── tests/
│   ├── test_grid.py                 # Coordinate system tests (50 tests)
│   ├── test_board.py                # Board rendering tests (8 tests)
│   ├── test_components.py           # Component and wire tests (11 tests)
│   └── test_mutations.py            # Mutation engine tests (15 tests)
├── reference/
│   └── wb102_photo.jpeg             # Real WB-102 photo (visual reference only)
└── output/                          # Generated images and datasets
```

## How It Works

The pipeline flows through six layers, each with a single responsibility:

```
board_spec.json          Define all physical dimensions
       ↓
    grid.py              Convert (row, col) → pixel coordinates
       ↓
 board.py + holes.py     Render blank board with holes, rails, labels
       ↓
components.py + wires.py Place resistors, LEDs, and wires on the board
       ↓
   mutations.py          Inject errors into circuit configs (for "incorrect" images)
       ↓
    augment.py           Apply realistic visual distortions (warp, blur, noise, etc.)
       ↓
   generate.py           Orchestrate batch output with automatic labels
```

**Labels are automatic** because mutations are tracked. When the pipeline injects a wrong wire connection, it records exactly which wire, which endpoint, and what the original vs. mutated positions were. This mutation record becomes the label metadata — no manual annotation needed.

## Circuit Config Format

Circuit configs are JSON files that define what components and wires to place on the board.

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Circuit identifier (used in labels) |
| `description` | string | No | Human-readable description |
| `components` | array | Yes | List of component definitions |
| `wires` | array | Yes | List of wire definitions |

### Component: Resistor

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (e.g., `"R1"`) |
| `type` | string | Yes | Must be `"resistor"` |
| `value` | string | No | Resistance value label (e.g., `"220"`, `"1k"`) |
| `pins.leg1` | [row, col] | Yes | First leg position (e.g., `["a", 10]`) |
| `pins.leg2` | [row, col] | Yes | Second leg position (e.g., `["a", 15]`) |
| `bands` | array | No | List of RGB color tuples for resistance bands. Defaults to brown-black-brown-gold (100 ohm). |

### Component: LED

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (e.g., `"LED1"`) |
| `type` | string | Yes | Must be `"led"` |
| `color` | string | No | LED color: `"red"`, `"green"`, `"yellow"`, `"blue"`, `"white"`. Defaults to `"red"`. |
| `pins.anode` | [row, col] | Yes | Anode (positive) leg position |
| `pins.cathode` | [row, col] | Yes | Cathode (negative) leg position |

### Wire

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `from` | [row, col] | Yes | Start position. Row can be `"a"`-`"j"` or a rail ID (`"p1+"`, `"p1-"`, `"p2+"`, `"p2-"`). |
| `to` | [row, col] | Yes | End position (same format as `from`) |
| `color` | string | No | Wire color: `"red"`, `"black"`, `"yellow"`, `"green"`, `"blue"`, `"white"`, `"orange"`. Defaults to `"red"`. |
| `routing` | string | No | `"straight"` (default) or `"L"` for L-shaped routing |

### Position reference

- **Terminal rows:** `"a"` through `"j"` (top half: a-e, bottom half: f-j)
- **Power rails:** `"p1+"` (top positive), `"p1-"` (top negative), `"p2-"` (bottom negative), `"p2+"` (bottom positive)
- **Columns:** 1 through 63

## Mutation Types

| Mutation | Description | Example |
|----------|-------------|---------|
| `remove_component` | Removes a random component entirely | Resistor R1 is removed from the board |
| `wrong_position` | Shifts a component's pins to nearby wrong columns | LED1 shifted 3 columns to the right |
| `wrong_connection` | Moves one end of a wire to a wrong column | Wire endpoint moved from column 20 to column 16 |
| `swap_polarity` | Swaps a +/- rail connection | Wire connected to p1+ is moved to p1- |
| `extra_component` | Adds a spurious resistor or LED | An extra LED appears at an unused position |
| `compound_mutation` | Applies 2+ mutations for harder examples | Wrong position + removed component together |

Each mutation returns a `mutation_record` dict documenting exactly what changed, which is saved in `labels.json`.

## Augmentation Types

| Augmentation | Parameters | Description |
|-------------|------------|-------------|
| `perspective_warp` | `max_angle_deg` (default: 15) | Simulates camera not being perfectly overhead |
| `lighting_jitter` | `brightness_range` ([-30, 30]), `contrast_range` ([0.8, 1.2]) | Random brightness and contrast shifts |
| `gaussian_blur` | `kernel_range` ([1, 5]) | Simulates slight out-of-focus |
| `rotation` | `max_angle_deg` (default: 5) | Slight in-plane rotation |
| `add_shadow` | `shadow_intensity_range` ([0.3, 0.7]) | Gradient shadow overlay (horizontal, vertical, or diagonal) |
| `add_noise` | `noise_std_range` ([5, 20]) | Gaussian sensor noise |
| `background_variation` | `bg_colors` (list of RGB values) | Replaces background color to simulate different surfaces |

Each augmentation can be individually enabled/disabled in `config/augmentation_config.json`. By default, 2-4 random augmentations are applied per image.

## Output Format

```
output/dataset_v1/
├── images/
│   ├── correct_0000.png
│   ├── correct_0001.png
│   ├── incorrect_0000.png
│   ├── incorrect_0001.png
│   └── ...
├── labels.json              # Per-image labels and metadata
├── metadata.json            # Generation parameters and timing
└── validation_report.txt    # Validation results
```

### labels.json schema

Each entry in the labels array:

```json
{
  "filename": "incorrect_0023.png",
  "label": "incorrect",
  "circuit_config": "simple_led_circuit",
  "mutations": [
    {
      "type": "wrong_connection",
      "wire_index": 2,
      "end_modified": "from",
      "original": ["b", 20],
      "mutated": ["b", 16]
    }
  ],
  "augmentations": ["rotation", "add_noise"],
  "augmentation_seed": 115,
  "seed": 42
}
```

For correct images, `mutations` is an empty array and `label` is `"correct"`.

### metadata.json

Records generation parameters: circuit config path, image counts, master seed, timestamp, and elapsed time.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_grid.py -v

# Run a specific test class
pytest tests/test_mutations.py::TestSwapPolarity -v
```

**84 tests** covering:

| Test file | Count | Coverage |
|-----------|-------|----------|
| `test_grid.py` | 50 | Coordinate conversion, hole spacing, rail grouping, connectivity, edge cases |
| `test_board.py` | 8 | Board dimensions, holes, stripes, center gap, body color |
| `test_components.py` | 11 | Wire drawing, resistor rendering, LED rendering |
| `test_mutations.py` | 15 | All 6 mutation types, immutability, reproducibility |

## Configuration Reference

### board_spec.json

Controls all physical dimensions of the WB-102 breadboard. Key tunable values:

| Key | Default | What it controls |
|-----|---------|-----------------|
| `board.pixels_per_mm` | 4.0 | Output resolution. Increase for sharper images. |
| `board.body_color` | [255, 255, 255] | Board body color (white) |
| `rendering.supersample_factor` | 2 | Render at Nx then downscale for antialiasing |
| `rendering.background_color` | [180, 180, 180] | Desk/surface color behind the board |
| `component_defaults.wire_colors` | {...} | Named wire color palette |
| `component_defaults.led_colors` | {...} | LED dome color palette |

The board dimensions (`real_width_mm`, `real_height_mm`, `holes.pitch_mm`, etc.) match the real WB-102 hardware and generally should not be changed.

### augmentation_config.json

Controls the intensity and probability of each augmentation. Each augmentation has an `enabled` flag and type-specific parameter ranges. The `apply_random.n_augmentations` field controls how many augmentations are applied per image (default: [2, 4]).

## Adding New Components

To add a new component type (e.g., capacitor, IC chip):

1. **Add a class** in `generator/components.py` following the existing pattern:

```python
class Capacitor:
    def __init__(self, pin1: tuple[str, int], pin2: tuple[str, int], value: str = "100nF"):
        self.pin1 = pin1
        self.pin2 = pin2
        self.value = value

    def draw(self, img: Image.Image, grid: BreadboardGrid) -> Image.Image:
        # Use grid.hole_center() for all positioning — no pixel math
        x1, y1 = grid.hole_center(self.pin1[0], self.pin1[1])
        x2, y2 = grid.hole_center(self.pin2[0], self.pin2[1])
        # ... draw the component ...
        return img
```

2. **Register it** in `generator/circuit.py` by adding a case to `_build_component()`:

```python
elif ctype == 'capacitor':
    pins = comp['pins']
    return Capacitor(pin1=_pin(pins['pin1']), pin2=_pin(pins['pin2']), value=comp.get('value', '100nF'))
```

3. **Add default dimensions** to `board_spec.json` under `component_defaults` (e.g., `capacitor_body_diameter_mm`).

4. **Use it** in a circuit config JSON:

```json
{ "id": "C1", "type": "capacitor", "value": "100nF", "pins": { "pin1": ["a", 30], "pin2": ["b", 30] } }
```

That's all for now... 
