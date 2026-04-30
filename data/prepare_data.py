"""
Phase 8a — CycleGAN data preparation.

Builds the two-domain training set used to train the synthetic→real CycleGAN:

  Domain A (synthetic): rendered images from this project's pipeline.
    Generated as a diverse mix so the GAN sees more than just three
    canned circuits — blank boards, wires-only layouts, resistor/LED
    builds, and Arduino/Metro-Mini layouts with assorted extras. Each
    image uses random component placements, random wire colours and
    routings, and a random subset gets mutated for further variety.
    Light augmentation only (slight rotation + lighting jitter) so
    CycleGAN learns realistic texture/lighting rather than memorising
    aggressive augmentation artefacts.

  Domain B (real): photos of physical WB-102 breadboards supplied by the user.

Both domains are resized to a single square resolution (default 256x256;
pass --image-size 512 to match a 512x512 CycleGAN run) using the same
letterbox routine, and split 80/20 train/test with a seeded shuffle.

Usage:
    python data/prepare_data.py \
        --real-source path/to/raw_photos/ \
        --spec config/board_spec.json \
        --out-dir data/ \
        --n-synthetic 500 \
        --image-size 256 \
        --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Allow running this script directly: `python data/prepare_data.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator.augment import AugmentationPipeline
from generator.board import draw_board_base
from generator.components import LED, Arduino, Resistor
from generator.grid import BreadboardGrid, load_spec
from generator.holes import draw_holes
from generator.mutations import MutationEngine
from generator.wires import draw_wire


SUPPORTED_PHOTO_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp', '.tiff'}

# Default category counts. Total = 500. --n-synthetic scales these
# proportionally if it differs from this base total.
DEFAULT_MIX = {
    'blank': 50,
    'wires_only': 100,
    'resistors_leds': 150,
    'arduino_plus': 200,
}

WIRE_COLOR_NAMES = ['red', 'black', 'yellow', 'green', 'blue', 'white', 'orange']
LED_COLOR_NAMES = ['red', 'green', 'yellow', 'blue', 'white']
TOP_ROWS = ['a', 'b', 'c', 'd', 'e']
BOTTOM_ROWS = ['f', 'g', 'h', 'i', 'j']
RAIL_IDS = ['p1+', 'p1-', 'p2+', 'p2-']

# Light augmentation: only slight rotation + minor lighting jitter.
# Everything else (perspective warp, blur, shadows, noise, background swap)
# is intentionally disabled — the CycleGAN will learn realistic appearance
# from the real photo domain.
LIGHT_AUGMENTATION_CONFIG = {
    "perspective_warp": {"enabled": False, "max_angle_deg": 0},
    "lighting_jitter": {
        "enabled": True,
        "brightness_range": [-12, 12],
        "contrast_range": [0.92, 1.08],
    },
    "gaussian_blur": {"enabled": False, "kernel_range": [1, 1]},
    "rotation": {"enabled": True, "max_angle_deg": 3},
    "add_shadow": {"enabled": False, "shadow_intensity_range": [0, 0]},
    "add_noise": {"enabled": False, "noise_std_range": [0, 0]},
    "background_variation": {"enabled": False, "bg_colors": [[180, 180, 180]]},
    "apply_random": {"n_augmentations": [1, 2]},
}


# ---------------------------------------------------------------------------
# Square resize (shared by synthetic + real for consistency)
# ---------------------------------------------------------------------------

# Fixed neutral grey used to pad the shorter axis when fitting a landscape
# image into a square canvas. Both synthetic renders and real photos use
# the same colour so the CycleGAN discriminator can't tell the domains
# apart by their padding bars — corner-sampled colours used to vary
# across real photos (desk / paper / sky) while staying constant in
# synthetic, which subtly biased training.
SQUARE_PAD_COLOR = (128, 128, 128)


def _square_resize(img: Image.Image, size: int) -> Image.Image:
    """Letterbox-resize to a (size, size) square. Used for BOTH domains."""
    img = img.convert('RGB')
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new('RGB', (size, size), SQUARE_PAD_COLOR)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


# ---------------------------------------------------------------------------
# Random circuit construction
# ---------------------------------------------------------------------------


def _random_wire(rng: random.Random, n_cols: int, allow_rail: bool = True) -> dict:
    """Build a random wire spec usable by draw_wire()."""
    pool = list(TOP_ROWS) + list(BOTTOM_ROWS)
    if allow_rail:
        pool = pool + list(RAIL_IDS)
    return {
        'from': [rng.choice(pool), rng.randint(1, n_cols)],
        'to': [rng.choice(pool), rng.randint(1, n_cols)],
        'color': rng.choice(WIRE_COLOR_NAMES),
        'routing': rng.choices(['straight', 'L'], weights=[0.6, 0.4])[0],
    }


def _random_resistor(rng: random.Random, n_cols: int) -> dict:
    """Resistor laid horizontally in one terminal half, ~3-7 cols across."""
    half = rng.choice([TOP_ROWS, BOTTOM_ROWS])
    row = rng.choice(half)
    span = rng.randint(3, 7)
    col = rng.randint(1, max(1, n_cols - span))
    bands = [
        (rng.randint(20, 230), rng.randint(20, 230), rng.randint(20, 230))
        for _ in range(4)
    ]
    return {
        'id': f'R{rng.randint(1, 9999)}',
        'type': 'resistor',
        'pins': {'leg1': [row, col], 'leg2': [row, col + span]},
        'bands': bands,
    }


def _random_led(rng: random.Random, n_cols: int) -> dict:
    """LED with anode/cathode in adjacent rows of the same terminal half."""
    half = rng.choice([TOP_ROWS, BOTTOM_ROWS])
    idx = rng.randint(0, len(half) - 2)
    anode_row, cathode_row = half[idx], half[idx + 1]
    if rng.random() < 0.5:
        anode_row, cathode_row = cathode_row, anode_row
    col = rng.randint(1, n_cols)
    return {
        'id': f'LED{rng.randint(1, 9999)}',
        'type': 'led',
        'color': rng.choice(LED_COLOR_NAMES),
        'pins': {'anode': [anode_row, col], 'cathode': [cathode_row, col]},
    }


def _random_arduino(rng: random.Random, spec: dict) -> dict:
    """Arduino spec for the Arduino component class."""
    defaults = spec['component_defaults']['arduino']
    n_pins = rng.randint(max(8, defaults['default_n_pins'] - 5),
                         defaults['default_n_pins'])
    n_cols = spec['terminal_strip']['columns']
    start_col = rng.randint(1, max(1, n_cols - n_pins))
    return {
        'start_col': start_col,
        'n_pins': n_pins,
        'top_row': defaults['default_top_row'],
        'bottom_row': defaults['default_bottom_row'],
        'usb_end': rng.choice(['left', 'right']),
    }


# ---------------------------------------------------------------------------
# In-memory rendering (no JSON round-trip needed for the random circuits)
# ---------------------------------------------------------------------------


def _render_scene(
    spec: dict,
    components: list[dict],
    wires: list[dict],
    arduinos: list[dict],
) -> Image.Image:
    """Render a full scene (board + holes + components + arduinos + wires).

    Renders at supersample resolution then downscales to base ppmm so the
    output matches the rest of the pipeline.
    """
    ss = spec['rendering']['supersample_factor']
    base_ppmm = spec['board']['pixels_per_mm']
    grid_hi = BreadboardGrid(spec, ppmm_override=base_ppmm * ss)

    img = draw_board_base(grid_hi)
    img = draw_holes(img, grid_hi)

    # Components first so wires sit on top (matches real-world appearance
    # where wires run over the body of nearby components).
    for c in components:
        if c['type'] == 'resistor':
            obj = Resistor(
                leg1=tuple(c['pins']['leg1']),
                leg2=tuple(c['pins']['leg2']),
                bands=[tuple(b) for b in c['bands']] if 'bands' in c else None,
            )
        elif c['type'] == 'led':
            obj = LED(
                anode=tuple(c['pins']['anode']),
                cathode=tuple(c['pins']['cathode']),
                color=c.get('color', 'red'),
            )
        else:
            raise ValueError(f"Unknown component type: {c['type']!r}")
        img = obj.draw(img, grid_hi)

    for a in arduinos:
        obj = Arduino(
            start_col=a['start_col'],
            n_pins=a.get('n_pins'),
            top_row=a.get('top_row'),
            bottom_row=a.get('bottom_row'),
            usb_end=a.get('usb_end', 'left'),
        )
        img = obj.draw(img, grid_hi)

    for w in wires:
        img = draw_wire(
            img, grid_hi,
            start=tuple(w['from']),
            end=tuple(w['to']),
            color=w['color'],
            routing=w['routing'],
        )

    if ss > 1:
        target_w = round(spec['board']['real_width_mm'] * base_ppmm)
        target_h = round(spec['board']['real_height_mm'] * base_ppmm)
        img = img.resize((target_w, target_h), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Per-category scene builders
# ---------------------------------------------------------------------------


def _maybe_mutate(circuit: dict, rng: random.Random, seed: int) -> tuple[dict, str]:
    """With ~50% probability, apply a random mutation to a circuit dict.

    Returns the (possibly mutated) circuit and a short tag describing kind.
    """
    if rng.random() >= 0.5 or not circuit.get('components') and not circuit.get('wires'):
        return circuit, 'clean'
    mutations = [
        'remove_component', 'wrong_position', 'wrong_connection',
        'swap_polarity', 'extra_component',
    ]
    eng = MutationEngine(seed=seed)
    chosen = rng.choice(mutations)
    try:
        mutated, _ = getattr(eng, chosen)(circuit)
        return mutated, f'mut_{chosen}'
    except (ValueError, IndexError, StopIteration):
        return circuit, 'clean'


def _build_blank() -> dict:
    return {'components': [], 'wires': []}


def _build_wires_only(rng: random.Random, n_cols: int) -> dict:
    n = rng.randint(3, 12)
    return {
        'components': [],
        'wires': [_random_wire(rng, n_cols) for _ in range(n)],
    }


def _build_resistors_leds(rng: random.Random, n_cols: int) -> dict:
    n_resistors = rng.randint(1, 4)
    n_leds = rng.randint(1, 4)
    n_wires = rng.randint(2, 8)
    components = (
        [_random_resistor(rng, n_cols) for _ in range(n_resistors)]
        + [_random_led(rng, n_cols) for _ in range(n_leds)]
    )
    return {
        'components': components,
        'wires': [_random_wire(rng, n_cols) for _ in range(n_wires)],
    }


def _build_arduino_plus(rng: random.Random, spec: dict) -> tuple[dict, list[dict]]:
    """Arduino + a mix of resistors / LEDs / wires. Returns (circuit, arduinos)."""
    n_cols = spec['terminal_strip']['columns']
    arduino = _random_arduino(rng, spec)

    # Components avoid the columns under the Arduino (visual realism).
    arduino_cols = set(range(arduino['start_col'],
                             arduino['start_col'] + arduino['n_pins']))

    n_resistors = rng.randint(0, 3)
    n_leds = rng.randint(1, 4)
    n_wires = rng.randint(3, 10)

    def _outside_arduino_resistor() -> dict | None:
        for _ in range(8):
            r = _random_resistor(rng, n_cols)
            cols = {r['pins']['leg1'][1], r['pins']['leg2'][1]}
            if not cols & arduino_cols:
                return r
        return None

    def _outside_arduino_led() -> dict | None:
        for _ in range(8):
            led = _random_led(rng, n_cols)
            if led['pins']['anode'][1] not in arduino_cols:
                return led
        return None

    components: list[dict] = []
    for _ in range(n_resistors):
        r = _outside_arduino_resistor()
        if r is not None:
            components.append(r)
    for _ in range(n_leds):
        led = _outside_arduino_led()
        if led is not None:
            components.append(led)

    wires = [_random_wire(rng, n_cols) for _ in range(n_wires)]
    return {'components': components, 'wires': wires}, [arduino]


# ---------------------------------------------------------------------------
# Mix scaling and synthetic generation
# ---------------------------------------------------------------------------


def _scale_mix(target_total: int) -> dict:
    """Scale DEFAULT_MIX so the category counts sum to ~target_total."""
    base_total = sum(DEFAULT_MIX.values())
    if target_total == base_total:
        return dict(DEFAULT_MIX)
    if target_total <= 0:
        return {k: 0 for k in DEFAULT_MIX}

    scale = target_total / base_total
    scaled = {k: max(1, round(v * scale)) for k, v in DEFAULT_MIX.items()}

    # Adjust the largest bucket to absorb any rounding error so the total
    # matches target_total exactly.
    delta = target_total - sum(scaled.values())
    if delta != 0:
        biggest = max(scaled, key=scaled.get)
        scaled[biggest] = max(1, scaled[biggest] + delta)
    return scaled


def collect_real_photos(source_dir: Path) -> list[Path]:
    """Return all supported image files under source_dir (non-recursive)."""
    if not source_dir.exists():
        return []
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_PHOTO_EXTS
    )


def process_real_photos(
    source_dir: Path,
    train_dir: Path,
    test_dir: Path,
    image_size: int,
    seed: int,
) -> dict:
    """Resize + 80/20 split real photos. Uses the same _square_resize as syn."""
    photos = collect_real_photos(source_dir)
    if not photos:
        return {"found": 0, "train": 0, "test": 0, "source": str(source_dir)}

    rng = random.Random(seed)
    shuffled = photos.copy()
    rng.shuffle(shuffled)

    n_train = int(round(len(shuffled) * 0.8))
    train_photos = shuffled[:n_train]
    test_photos = shuffled[n_train:]

    for split_dir, items in ((train_dir, train_photos), (test_dir, test_photos)):
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx, src in enumerate(items):
            try:
                with Image.open(src) as im:
                    im.load()
                    out = _square_resize(im, image_size)
            except Exception as e:
                print(f"  WARN: skipping {src.name}: {e}")
                continue
            out_name = f"real_{idx:04d}.png"
            out.save(split_dir / out_name)

    return {
        "found": len(photos),
        "train": len(train_photos),
        "test": len(test_photos),
        "source": str(source_dir),
    }


def generate_synthetic_set(
    spec: dict,
    n_images: int,
    image_size: int,
    seed: int,
    train_dir: Path,
    test_dir: Path,
) -> dict:
    """
    Render n_images synthetic breadboards as a diverse mix:
      - blank boards
      - wires-only boards
      - boards with resistors and LEDs
      - boards with an Arduino/Metro Mini plus assorted other components

    Random component placements, random wire colours/routing, and roughly
    half of the non-blank scenes get a random mutation applied for variety.
    Light augmentation only, then resized via _square_resize and split 80/20.
    """
    mix = _scale_mix(n_images)
    rng = random.Random(seed)
    n_cols = spec['terminal_strip']['columns']

    # Build a deterministic list of (category, scene_seed) tuples so the
    # output is reproducible for a given --seed.
    plan: list[str] = []
    for cat, count in mix.items():
        plan.extend([cat] * count)
    rng.shuffle(plan)

    images: list[tuple[str, Image.Image]] = []
    cat_counts = {k: 0 for k in DEFAULT_MIX}
    mutation_counts = {'clean': 0}

    for i, category in enumerate(plan):
        scene_seed = seed + i + 1
        scene_rng = random.Random(scene_seed)

        if category == 'blank':
            circuit = _build_blank()
            arduinos = []
        elif category == 'wires_only':
            circuit = _build_wires_only(scene_rng, n_cols)
            arduinos = []
        elif category == 'resistors_leds':
            circuit = _build_resistors_leds(scene_rng, n_cols)
            arduinos = []
        elif category == 'arduino_plus':
            circuit, arduinos = _build_arduino_plus(scene_rng, spec)
        else:
            raise ValueError(f"Unknown category: {category!r}")

        # Mutate ~50% of scenes that have content to mutate.
        if category == 'blank':
            kind = 'blank'
        else:
            circuit, kind = _maybe_mutate(circuit, scene_rng, scene_seed)
        mutation_counts[kind] = mutation_counts.get(kind, 0) + 1

        img = _render_scene(
            spec=spec,
            components=circuit.get('components', []),
            wires=circuit.get('wires', []),
            arduinos=arduinos,
        )

        aug = AugmentationPipeline(LIGHT_AUGMENTATION_CONFIG, seed=scene_seed)
        aug_img, _ = aug.apply_random_pil(img)
        aug_img = _square_resize(aug_img, image_size)

        stem = f"syn_{i:04d}_{category}_{kind}"
        images.append((stem, aug_img))
        cat_counts[category] += 1

    rng.shuffle(images)
    n_train = int(round(len(images) * 0.8))
    train_imgs = images[:n_train]
    test_imgs = images[n_train:]

    for split_dir, items in ((train_dir, train_imgs), (test_dir, test_imgs)):
        split_dir.mkdir(parents=True, exist_ok=True)
        for stem, im in items:
            im.save(split_dir / f"{stem}.png")

    return {
        "rendered": len(images),
        "train": len(train_imgs),
        "test": len(test_imgs),
        "mix": cat_counts,
        "mutation_kinds": mutation_counts,
    }


def _clean_split_dir(d: Path) -> None:
    """Remove existing PNG/JPG files in a split directory (keeps the dir)."""
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            p.unlink()


def prepare_data(
    real_source: Path | None,
    out_dir: Path,
    spec_path: Path,
    n_synthetic: int,
    image_size: int,
    seed: int,
    clean: bool,
) -> dict:
    """Run the full prep: synthetic generation + real photo processing.

    Both domains are resized to (image_size, image_size) using the same
    _square_resize helper so the GAN sees a consistent target resolution.
    """
    spec = load_spec(str(spec_path))

    syn_train = out_dir / 'synthetic' / 'train'
    syn_test = out_dir / 'synthetic' / 'test'
    real_train = out_dir / 'real' / 'train'
    real_test = out_dir / 'real' / 'test'

    for d in (syn_train, syn_test, real_train, real_test):
        d.mkdir(parents=True, exist_ok=True)
        if clean:
            _clean_split_dir(d)

    print(f"Generating {n_synthetic} synthetic images "
          f"(diverse mix; light aug; {image_size}x{image_size})...")
    syn_stats = generate_synthetic_set(
        spec=spec,
        n_images=n_synthetic,
        image_size=image_size,
        seed=seed,
        train_dir=syn_train,
        test_dir=syn_test,
    )

    if real_source is not None:
        print(f"Processing real photos from {real_source} "
              f"(letterboxed to {image_size}x{image_size})...")
        real_stats = process_real_photos(
            source_dir=real_source,
            train_dir=real_train,
            test_dir=real_test,
            image_size=image_size,
            seed=seed,
        )
    else:
        real_stats = {
            "found": 0, "train": 0, "test": 0,
            "source": "(not provided — drop photos in data/real/ later "
                      "and rerun with --real-source)",
        }

    stats = {
        "image_size": image_size,
        "seed": seed,
        "synthetic": syn_stats,
        "real": real_stats,
        "out_dir": str(out_dir),
    }
    return stats


def _print_stats(stats: dict) -> None:
    size = stats['image_size']
    print()
    print("=" * 60)
    print("Data preparation summary")
    print("=" * 60)
    print(f"Resolution:        {size}x{size}")
    print(f"Seed:              {stats['seed']}")
    print(f"Output directory:  {stats['out_dir']}")
    print()
    syn = stats['synthetic']
    print("Domain A — synthetic")
    print(f"  rendered:        {syn['rendered']}")
    for cat, n in syn.get('mix', {}).items():
        print(f"    {cat:18s} {n}")
    print(f"  train split:     {syn['train']}")
    print(f"  test split:      {syn['test']}")
    print()
    real = stats['real']
    print("Domain B — real")
    print(f"  source:          {real['source']}")
    print(f"  found:           {real['found']}")
    print(f"  train split:     {real['train']}")
    print(f"  test split:      {real['test']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CycleGAN training data (synthetic + real domains).",
    )
    parser.add_argument(
        '--real-source', type=Path, default=None,
        help="Directory of raw real breadboard photos. If omitted, only the "
             "synthetic domain is generated and the real/ split is left empty.",
    )
    parser.add_argument(
        '--spec', type=Path, default=PROJECT_ROOT / 'config' / 'board_spec.json',
        help="Path to board_spec.json.",
    )
    parser.add_argument(
        '--out-dir', type=Path, default=PROJECT_ROOT / 'data',
        help="Top-level output directory containing synthetic/ and real/.",
    )
    parser.add_argument(
        '--n-synthetic', type=int, default=sum(DEFAULT_MIX.values()),
        help="Total synthetic images to generate. Default 500 spread across "
             "blank/wires/resistors-leds/arduino as 50/100/150/200. Different "
             "totals scale these counts proportionally.",
    )
    parser.add_argument(
        '--image-size', type=int, default=256,
        help="Square output resolution applied to BOTH synthetic and real "
             "(CycleGAN default 256; pass 512 to match a 512x512 train run).",
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Master random seed for shuffling and rendering.",
    )
    parser.add_argument(
        '--clean', action='store_true',
        help="Delete existing images in the output split directories before writing.",
    )
    parser.add_argument(
        '--stats-out', type=Path, default=None,
        help="Optional path to write the stats summary as JSON.",
    )

    args = parser.parse_args()

    stats = prepare_data(
        real_source=args.real_source,
        out_dir=args.out_dir,
        spec_path=args.spec,
        n_synthetic=args.n_synthetic,
        image_size=args.image_size,
        seed=args.seed,
        clean=args.clean,
    )

    _print_stats(stats)

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_out, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats JSON written to {args.stats_out}")


if __name__ == '__main__':
    main()
