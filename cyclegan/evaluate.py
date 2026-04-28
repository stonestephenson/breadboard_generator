"""
CycleGAN structural-preservation evaluator — Phase 8c.

Translation is only useful for our pipeline if components stay in the same
pixel positions after stylisation. This script computes SSIM (structural
similarity) between each synthetic source image and its translated version
and flags images whose score falls below a threshold.

Usage:
    python -m cyclegan.evaluate \
        --source data/synthetic/test \
        --translated output/stylized \
        --threshold 0.6 \
        --report-csv cyclegan/logs/eval_ssim.csv

Bounding boxes are unaffected if SSIM stays high. Drift in component
position usually corresponds to a noticeable SSIM drop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def _load_resized(path: Path, size: int) -> np.ndarray:
    """Load an image as an HxWx3 float array in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert('RGB').resize((size, size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def _match_pairs(
    source_dir: Path, translated_dir: Path,
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """
    Find source/translated image pairs by stem.

    Returns (pairs, missing_translations). `pairs` is sorted by stem;
    `missing_translations` lists source stems that have no translated counterpart.
    """
    sources = {
        p.stem: p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    }
    targets = {
        p.stem: p for p in translated_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    }
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for stem in sorted(sources):
        if stem in targets:
            pairs.append((sources[stem], targets[stem]))
        else:
            missing.append(stem)
    return pairs, missing


def evaluate_directory(
    source_dir: Path,
    translated_dir: Path,
    image_size: int,
    threshold: float,
) -> tuple[list[dict], dict]:
    pairs, missing = _match_pairs(source_dir, translated_dir)
    if not pairs:
        raise RuntimeError(
            f"No matching source/translated stems between {source_dir} and {translated_dir}."
        )

    rows: list[dict] = []
    for src_path, tgt_path in pairs:
        src = _load_resized(src_path, image_size)
        tgt = _load_resized(tgt_path, image_size)

        # Multichannel SSIM. data_range=1.0 because both arrays are in [0, 1].
        score = float(ssim(src, tgt, data_range=1.0, channel_axis=2))
        rows.append({
            'stem': src_path.stem,
            'source': str(src_path),
            'translated': str(tgt_path),
            'ssim': round(score, 4),
            'flagged': score < threshold,
        })

    scores = [r['ssim'] for r in rows]
    flagged = [r for r in rows if r['flagged']]

    summary = {
        'n_pairs': len(rows),
        'n_missing_translations': len(missing),
        'missing': missing,
        'threshold': threshold,
        'mean_ssim': round(mean(scores), 4),
        'median_ssim': round(median(scores), 4),
        'stdev_ssim': round(stdev(scores), 4) if len(scores) > 1 else 0.0,
        'min_ssim': round(min(scores), 4),
        'max_ssim': round(max(scores), 4),
        'n_flagged': len(flagged),
    }
    return rows, summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['stem', 'ssim', 'flagged', 'source', 'translated']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def _print_summary(summary: dict, rows: list[dict], top_n: int = 10) -> None:
    print()
    print("=" * 60)
    print("CycleGAN structural-preservation evaluation")
    print("=" * 60)
    print(f"Pairs scored:           {summary['n_pairs']}")
    print(f"Missing translations:   {summary['n_missing_translations']}")
    print(f"SSIM threshold:         {summary['threshold']}")
    print(f"Mean SSIM:              {summary['mean_ssim']}")
    print(f"Median SSIM:            {summary['median_ssim']}")
    print(f"Stdev SSIM:             {summary['stdev_ssim']}")
    print(f"Min / Max SSIM:         {summary['min_ssim']} / {summary['max_ssim']}")
    print(f"Flagged (< threshold):  {summary['n_flagged']}")

    if summary['n_flagged']:
        worst = sorted(rows, key=lambda r: r['ssim'])[:top_n]
        print(f"\nWorst {len(worst)} images:")
        for r in worst:
            print(f"  {r['ssim']:.4f}  {r['stem']}")

    if summary['missing']:
        print(f"\nFirst few missing translations:")
        for stem in summary['missing'][:top_n]:
            print(f"  - {stem}")

    print("=" * 60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute SSIM between synthetic sources and CycleGAN translations.",
    )
    p.add_argument('--source', type=Path, required=True,
                   help="Directory of synthetic source images.")
    p.add_argument('--translated', type=Path, required=True,
                   help="Directory of CycleGAN-translated images (matched by filename stem).")
    p.add_argument('--image-size', type=int, default=256,
                   help="Resize both sides to this square resolution before SSIM.")
    p.add_argument('--threshold', type=float, default=0.6,
                   help="Flag images whose SSIM is below this value.")
    p.add_argument('--report-csv', type=Path, default=None,
                   help="Optional path to write a per-image CSV report.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rows, summary = evaluate_directory(
        source_dir=args.source,
        translated_dir=args.translated,
        image_size=args.image_size,
        threshold=args.threshold,
    )
    _print_summary(summary, rows)

    if args.report_csv:
        write_csv(rows, args.report_csv)
        print(f"\nPer-image CSV written to {args.report_csv}")


if __name__ == '__main__':
    main()
