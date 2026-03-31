# CLAUDE.md — Breadboard Synthetic Data Generator V2

## What This Project Does

Generates synthetic training images of breadboard circuits for a CV model that classifies student-built circuits as correct/incorrect. Images are programmatically rendered with Pillow/OpenCV and automatically labeled.

## Reference Photo

`reference/wb102_photo.jpg` — A real photo of the WB-102 with components on it. Use for visual comparison only — do NOT extract measurements from this image. All dimensions come from `config/board_spec.json`.

**Key visual details confirmed from the reference photo:**
- Board body is **bright white** plastic, not off-white or cream
- Row labels are **lowercase** (`a b c d e`, `f g h i j`), printed at **both ends** of the board
- Row labels are printed **between the power rails and the terminal area**, not on the edge
- Column numbers are printed along the edge near the power rails
- Power rail stripes are **thin continuous red and blue lines** running the full board length
- Red `+` and blue `-` symbols are printed at the rail ends (red outside, blue inside)
- Power rails have a visible **gap/split near the center** of the board
- The center gap (DIP channel) is a **narrow recessed channel** — not a wide painted bar
- Holes appear as small dark squares with slightly rounded corners
- The board in the photo is shown in portrait orientation, but labels confirm landscape is canonical

## Read First

**Read `ARCHITECTURE.md` before writing any code.** It contains the full system design, layer responsibilities, module APIs, and implementation order. Every design decision is documented there.

## Critical Design Rules

### 1. Single Source of Truth
ALL physical dimensions come from `config/board_spec.json`. If you need a measurement, read it from the spec. Never hardcode pixel values in rendering code.

### 2. Pixel Math Isolation
`generator/grid.py` is the ONLY module that converts physical coordinates to pixel coordinates. Every other module calls `grid.hole_center(row, col)` or similar. If you find yourself writing `x = col * pitch + offset` anywhere other than `grid.py`, you're doing it wrong.

### 3. Validation Over Visual Inspection
Every rendered image must pass automated validation checks in `generator/validate.py`. The human should never need to "eyeball" output to know if it's correct. Write the validation check BEFORE the rendering code when possible.

### 4. Config Over Code
Changing which circuit to render = change a JSON file, not Python code. Changing board dimensions = change `board_spec.json`, not rendering code. Changing augmentation intensity = change `augmentation_config.json`.

### 5. Reproducibility
Seed all randomness. `numpy.random.seed()` and `random.seed()` must be set from a configurable seed passed through the pipeline. Same seed + same config = identical output, always.

## Board Spec Quick Reference

Target model: **WB-102 Solderless Breadboard by Circuit Specialists** (standard 830-point breadboard).

- Overall: 165.1 x 54.0 mm (6.5" x 2.125") — **landscape** orientation
- Hole pitch: 2.54mm (0.1") — universal electronics standard
- Hole diameter: ~1.0mm, slightly square with rounded corners
- Terminal area: 63 columns × 10 rows (A-E top, F-J bottom)
- Center gap: 7.62mm (0.3") — for DIP IC packages
- Power rails: 4 total (2 top, 2 bottom), each split into two 25-hole segments
- Rail markings: red stripe (+) on outside, blue stripe (-) on inside
- Column labels every 5 cols (1, 5, 10, 15, ... 60) plus 63
- 830 total tie-points: 630 terminal + 200 power rail
- Body: white/off-white ABS plastic

At `pixels_per_mm = 4.0`:
- Board: ~660px × 216px (landscape)
- Hole pitch: ~10px
- Hole diameter: ~4px
- Center gap: ~30px

## Module Responsibilities

| Module | Responsibility | Key Rule |
|--------|---------------|----------|
| `config/board_spec.json` | All physical dimensions | Only source of measurements |
| `generator/grid.py` | Coordinate conversion (row,col) → (px,py) | Only module doing pixel math |
| `generator/board.py` | Draw board base, gap, labels | Reads grid.py for positions |
| `generator/holes.py` | Draw hole grid | Reads grid.py for positions |
| `generator/components.py` | Draw electronic components | Each component is a class |
| `generator/wires.py` | Draw wire connections | Supports straight/L/bezier |
| `generator/circuit.py` | Load circuit JSON, orchestrate render | No drawing logic here |
| `generator/mutations.py` | Inject errors into circuits | Returns mutation metadata |
| `generator/augment.py` | OpenCV post-processing | Parameterized, seeded |
| `generator/validate.py` | Automated correctness checks | Runs after every render |
| `generate.py` | Batch orchestration | Main entry point |

## Implementation Order

Build in this exact order. Each phase must pass its tests before moving on.

1. **Spec + Grid + Blank Board** → `board_spec.json`, `grid.py`, `board.py`, `holes.py`, `test_grid.py`
2. **Components** → `components.py`, `wires.py`
3. **Circuit Configs** → `circuit.py` + example JSON configs
4. **Mutations** → `mutations.py`, `test_mutations.py`
5. **Augmentation** → `augment.py`, `augmentation_config.json`
6. **Batch Generation** → `generate.py` with full validation

## Tech Stack

- **Python 3.10+**
- **Pillow** — 2D rendering (board, components)
- **OpenCV** (`opencv-python`) — augmentation (warp, blur, lighting)
- **NumPy** — array operations, image conversion between Pillow/OpenCV
- **pytest** — testing

## Conventions

- Type hints on all function signatures
- Docstrings on all public functions
- Snake_case for functions/variables, PascalCase for classes
- All configs are JSON (not YAML, not Python dicts)
- Test files mirror source files: `generator/grid.py` → `tests/test_grid.py`
- Output images are PNG format
- Labels are JSON format

## Common Mistakes to Avoid

- **Don't compute pixel positions in rendering code.** Call `grid.hole_center()`.
- **Don't hardcode colors.** Read them from `board_spec.json`.
- **Don't assume board orientation.** The spec defines it; the renderer reads it.
- **Don't skip validation.** Every `generate` call should run validation by default.
- **Don't use `random` without seeding.** Every function that uses randomness takes a `seed` or `rng` parameter.
- **Don't render at a fixed resolution.** Use `pixels_per_mm` from the spec. Higher resolution = change one number.
- **Don't test by looking at images.** Write assertions in `validate.py` and `tests/`.

## Running

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Generate single test image (Phase 1+)
python -c "from generator.board import render_blank_board; render_blank_board('config/board_spec.json', 'output/test.png')"

# Generate full dataset (Phase 6)
python generate.py --circuit config/circuits/simple_led.json --n-correct 1000 --n-incorrect 4000 --seed 42 --output output/dataset_v1/
```

## Git Discipline

- Run `git add -A && git commit` after EVERY change that passes tests.
- Use descriptive commit messages (e.g., "Fix rail hole vertical alignment").
- NEVER make multiple unrelated changes between commits.
- Before making any risky change, commit the current working state first.