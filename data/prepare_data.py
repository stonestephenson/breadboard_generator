"""
Phase 8a — CycleGAN data preparation.

Builds the two-domain training set used to train the synthetic→real CycleGAN:

  Domain A (synthetic): rendered images from this project's pipeline,
    augmented only lightly so CycleGAN learns the realistic texture/lighting
    rather than memorising aggressive augmentation artefacts.

  Domain B (real): photos of physical WB-102 breadboards supplied by the user.

Both domains are resized to a single square resolution (default 256x256)
and split 80/20 train/test with a seeded shuffle.

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

# Allow running this script directly: `python data/prepare_data.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator.augment import AugmentationPipeline
from generator.board import draw_board_base
from generator.circuit import load_circuit, render_circuit
from generator.grid import BreadboardGrid, load_spec
from generator.holes import draw_holes
from generator.mutations import MutationEngine


SUPPORTED_PHOTO_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp', '.tiff'}

DEFAULT_CIRCUIT_CONFIGS = [
    'config/circuits/simple_led.json',
    'config/circuits/dual_led.json',
    'config/circuits/resistor_divider.json',
]

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


def _render_blank(spec: dict) -> Image.Image:
    """Render a clean breadboard with no components, downscaled to target size."""
    ss = spec['rendering']['supersample_factor']
    base_ppmm = spec['board']['pixels_per_mm']
    grid_hi = BreadboardGrid(spec, ppmm_override=base_ppmm * ss)

    img = draw_board_base(grid_hi)
    img = draw_holes(img, grid_hi)

    if ss > 1:
        target_w = round(spec['board']['real_width_mm'] * base_ppmm)
        target_h = round(spec['board']['real_height_mm'] * base_ppmm)
        img = img.resize((target_w, target_h), Image.LANCZOS)
    return img


def _render_circuit_downscaled(circuit_config: dict, spec: dict) -> Image.Image:
    """Render a circuit at supersample resolution and downscale to base ppmm."""
    ss = spec['rendering']['supersample_factor']
    base_ppmm = spec['board']['pixels_per_mm']

    img = render_circuit(circuit_config, spec, ppmm_override=base_ppmm * ss)
    if ss > 1:
        target_w = round(spec['board']['real_width_mm'] * base_ppmm)
        target_h = round(spec['board']['real_height_mm'] * base_ppmm)
        img = img.resize((target_w, target_h), Image.LANCZOS)
    return img


def _square_resize(img: Image.Image, size: int) -> Image.Image:
    """
    Resize to a square (size x size) without distorting aspect ratio.

    Pads the shorter axis with the image's edge background colour so the
    breadboard fits inside the square. CycleGAN trains on square crops,
    and our boards are landscape, so direct resize would squash them.
    """
    img = img.convert('RGB')
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Sample a corner pixel as the pad colour (board renders have a flat bg;
    # real photos: just use a mid-grey, which CycleGAN will see as background).
    pad_color = resized.getpixel((0, 0))
    canvas = Image.new('RGB', (size, size), pad_color)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


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
    """Resize + 80/20 split real photos. Returns stats dict."""
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
    circuit_configs: list[dict],
    n_images: int,
    image_size: int,
    seed: int,
    train_dir: Path,
    test_dir: Path,
) -> dict:
    """
    Render n_images synthetic breadboards using a mix of:
      - Each provided circuit config (clean)
      - Mutated versions of each circuit
      - Some blank board renders

    Applies light augmentation only, then resizes to image_size and
    writes an 80/20 train/test split.

    Returns stats dict.
    """
    rng = random.Random(seed)

    # ~10% of the set is blank boards for variety; the rest splits across
    # circuits with a mix of clean and mutated versions.
    n_blank = max(1, n_images // 10)
    n_circuit = n_images - n_blank

    mutation_types = [
        'remove_component',
        'wrong_position',
        'wrong_connection',
        'swap_polarity',
        'extra_component',
        'compound_mutation',
    ]

    images: list[tuple[str, Image.Image]] = []

    # Render circuit-based images (alternate clean / mutated).
    for i in range(n_circuit):
        circuit_idx = i % len(circuit_configs)
        circuit = circuit_configs[circuit_idx]
        img_seed = seed + i

        # Roughly half mutated, half clean.
        if i % 2 == 0:
            config_to_render = circuit
            kind = 'clean'
        else:
            mut = MutationEngine(seed=img_seed)
            mut_type = mutation_types[i % len(mutation_types)]
            try:
                mutated, _ = getattr(mut, mut_type)(circuit)
                config_to_render = mutated
                kind = f'mut_{mut_type}'
            except (ValueError, IndexError, StopIteration):
                config_to_render = circuit
                kind = 'clean'

        img = _render_circuit_downscaled(config_to_render, spec)
        aug = AugmentationPipeline(LIGHT_AUGMENTATION_CONFIG, seed=img_seed)
        aug_img, _ = aug.apply_random_pil(img)
        aug_img = _square_resize(aug_img, image_size)

        circuit_name = circuit.get('name', f'circuit{circuit_idx}')
        stem = f"syn_{i:04d}_{circuit_name}_{kind}"
        images.append((stem, aug_img))

    # Render blank boards with light aug.
    for j in range(n_blank):
        img_seed = seed + n_circuit + j
        img = _render_blank(spec)
        aug = AugmentationPipeline(LIGHT_AUGMENTATION_CONFIG, seed=img_seed)
        aug_img, _ = aug.apply_random_pil(img)
        aug_img = _square_resize(aug_img, image_size)
        stem = f"syn_{n_circuit + j:04d}_blank"
        images.append((stem, aug_img))

    # Shuffle then split 80/20.
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
        "blank_count": n_blank,
        "circuit_count": n_circuit,
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
    circuit_paths: list[Path],
    n_synthetic: int,
    image_size: int,
    seed: int,
    clean: bool,
) -> dict:
    """Run the full prep: synthetic generation + real photo processing."""
    spec = load_spec(str(spec_path))
    circuits = [load_circuit(str(p)) for p in circuit_paths]

    syn_train = out_dir / 'synthetic' / 'train'
    syn_test = out_dir / 'synthetic' / 'test'
    real_train = out_dir / 'real' / 'train'
    real_test = out_dir / 'real' / 'test'

    for d in (syn_train, syn_test, real_train, real_test):
        d.mkdir(parents=True, exist_ok=True)
        if clean:
            _clean_split_dir(d)

    print(f"Generating {n_synthetic} synthetic images "
          f"({len(circuits)} circuit configs + blanks, light aug)...")
    syn_stats = generate_synthetic_set(
        spec=spec,
        circuit_configs=circuits,
        n_images=n_synthetic,
        image_size=image_size,
        seed=seed,
        train_dir=syn_train,
        test_dir=syn_test,
    )

    if real_source is not None:
        print(f"Processing real photos from {real_source}...")
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
    print(f"    circuits:      {syn['circuit_count']}")
    print(f"    blank boards:  {syn['blank_count']}")
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
        '--circuit', type=Path, action='append', default=None,
        help="Circuit config JSON (repeatable). Defaults to all 3 example circuits.",
    )
    parser.add_argument(
        '--out-dir', type=Path, default=PROJECT_ROOT / 'data',
        help="Top-level output directory containing synthetic/ and real/.",
    )
    parser.add_argument(
        '--n-synthetic', type=int, default=500,
        help="Total synthetic images to generate before the train/test split.",
    )
    parser.add_argument(
        '--image-size', type=int, default=256,
        help="Square output resolution (CycleGAN default 256).",
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

    circuit_paths = args.circuit or [
        PROJECT_ROOT / p for p in DEFAULT_CIRCUIT_CONFIGS
    ]

    stats = prepare_data(
        real_source=args.real_source,
        out_dir=args.out_dir,
        spec_path=args.spec,
        circuit_paths=circuit_paths,
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
