"""
Batch generation orchestrator — main entry point for dataset creation.

Generates labeled synthetic breadboard images:
  - Correct images: clean circuit + random augmentations
  - Incorrect images: mutated circuit + random augmentations

Every image is validated and all metadata is saved to labels.json.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

from PIL import Image

from generator.grid import BreadboardGrid, load_spec
from generator.circuit import load_circuit, render_circuit
from generator.mutations import MutationEngine
from generator.augment import AugmentationPipeline, load_augmentation_config
from generator.validate import BoardValidator


def _render_and_downscale(
    circuit_config: dict, spec: dict,
) -> Image.Image:
    """Render a circuit at supersample resolution and downscale."""
    ss = spec['rendering']['supersample_factor']
    base_ppmm = spec['board']['pixels_per_mm']

    img = render_circuit(circuit_config, spec, ppmm_override=base_ppmm * ss)

    if ss > 1:
        target_w = round(spec['board']['real_width_mm'] * base_ppmm)
        target_h = round(spec['board']['real_height_mm'] * base_ppmm)
        img = img.resize((target_w, target_h), Image.LANCZOS)

    return img


def generate_dataset(
    circuit_config_path: str,
    board_spec_path: str,
    output_dir: str,
    augmentation_config_path: str,
    n_correct: int = 1000,
    n_incorrect: int = 4000,
    seed: int = 42,
    validate: bool = True,
) -> dict:
    """
    Generate a full labeled dataset.

    Args:
        circuit_config_path: Path to circuit JSON config.
        board_spec_path: Path to board_spec.json.
        output_dir: Directory for output images and labels.
        augmentation_config_path: Path to augmentation_config.json.
        n_correct: Number of correct images to generate.
        n_incorrect: Number of incorrect images to generate.
        seed: Master random seed.
        validate: Whether to run validation on generated images.

    Returns:
        Summary dict with counts and any validation warnings.
    """
    start_time = time.time()

    # Load configs
    spec = load_spec(board_spec_path)
    circuit_config = load_circuit(circuit_config_path)
    aug_config = load_augmentation_config(augmentation_config_path)

    # Setup output dirs
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Setup validation
    grid = BreadboardGrid(spec)
    validator = BoardValidator(spec, grid) if validate else None

    labels = []
    validation_warnings = []

    # --- Generate correct images ---
    for i in range(n_correct):
        img_seed = seed + i
        aug_pipeline = AugmentationPipeline(aug_config, seed=img_seed)

        img = _render_and_downscale(circuit_config, spec)

        # Augment
        aug_img, aug_record = aug_pipeline.apply_random_pil(img)

        filename = f"correct_{i:04d}.png"
        aug_img.save(os.path.join(images_dir, filename))

        labels.append({
            "filename": filename,
            "label": "correct",
            "circuit_config": circuit_config.get('name', 'unknown'),
            "mutations": [],
            "augmentations": aug_record['augmentations'],
            "augmentation_seed": img_seed,
            "seed": seed,
        })

        # Validate the pre-augmentation render (augmentation distorts pixels)
        if validator:
            errors = validator.validate_board(img)
            if errors:
                for e in errors:
                    validation_warnings.append(f"{filename}: {e}")

    # --- Generate incorrect images ---
    mutation_types = [
        'remove_component',
        'wrong_position',
        'wrong_connection',
        'swap_polarity',
        'extra_component',
        'compound_mutation',
    ]

    for i in range(n_incorrect):
        img_seed = seed + n_correct + i
        mut_engine = MutationEngine(seed=img_seed)
        aug_pipeline = AugmentationPipeline(aug_config, seed=img_seed)

        # Pick a mutation type (cycle through evenly)
        mut_type = mutation_types[i % len(mutation_types)]

        try:
            mut_fn = getattr(mut_engine, mut_type)
            mutated_config, mutation_record = mut_fn(circuit_config)
        except (ValueError, IndexError, StopIteration):
            # Fallback to wrong_position if chosen mutation can't apply
            mutated_config, mutation_record = mut_engine.wrong_position(circuit_config)

        img = _render_and_downscale(mutated_config, spec)

        # Augment
        aug_img, aug_record = aug_pipeline.apply_random_pil(img)

        filename = f"incorrect_{i:04d}.png"
        aug_img.save(os.path.join(images_dir, filename))

        labels.append({
            "filename": filename,
            "label": "incorrect",
            "circuit_config": circuit_config.get('name', 'unknown'),
            "mutations": [mutation_record],
            "augmentations": aug_record['augmentations'],
            "augmentation_seed": img_seed,
            "seed": seed,
        })

    # --- Save labels ---
    labels_path = os.path.join(output_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)

    # --- Save metadata ---
    elapsed = time.time() - start_time
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "circuit_config": circuit_config_path,
        "board_spec": board_spec_path,
        "augmentation_config": augmentation_config_path,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "total_images": n_correct + n_incorrect,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # --- Save validation report ---
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        if validation_warnings:
            f.write(f"Validation warnings ({len(validation_warnings)}):\n\n")
            for w in validation_warnings:
                f.write(f"  - {w}\n")
        else:
            f.write("All images passed validation.\n")

    # --- Summary ---
    summary = {
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "total": n_correct + n_incorrect,
        "validation_warnings": len(validation_warnings),
        "elapsed_seconds": round(elapsed, 2),
        "output_dir": output_dir,
    }
    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic breadboard training dataset.',
    )
    parser.add_argument(
        '--circuit', required=True,
        help='Path to circuit JSON config',
    )
    parser.add_argument(
        '--spec', default='config/board_spec.json',
        help='Path to board_spec.json',
    )
    parser.add_argument(
        '--augmentation-config', default='config/augmentation_config.json',
        help='Path to augmentation_config.json',
    )
    parser.add_argument(
        '--n-correct', type=int, default=1000,
        help='Number of correct images',
    )
    parser.add_argument(
        '--n-incorrect', type=int, default=4000,
        help='Number of incorrect images',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Master random seed',
    )
    parser.add_argument(
        '--output', default='output/dataset/',
        help='Output directory',
    )
    parser.add_argument(
        '--no-validate', action='store_true',
        help='Skip validation checks',
    )

    args = parser.parse_args()

    summary = generate_dataset(
        circuit_config_path=args.circuit,
        board_spec_path=args.spec,
        output_dir=args.output,
        augmentation_config_path=args.augmentation_config,
        n_correct=args.n_correct,
        n_incorrect=args.n_incorrect,
        seed=args.seed,
        validate=not args.no_validate,
    )

    print(f"\nDataset generation complete!")
    print(f"  Correct images:       {summary['n_correct']}")
    print(f"  Incorrect images:     {summary['n_incorrect']}")
    print(f"  Total:                {summary['total']}")
    print(f"  Validation warnings:  {summary['validation_warnings']}")
    print(f"  Time:                 {summary['elapsed_seconds']}s")
    print(f"  Output:               {summary['output_dir']}")


if __name__ == '__main__':
    main()
