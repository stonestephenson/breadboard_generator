"""
Crop breadboard photos so the breadboard fills the frame.

Detects the largest light-colored rectangular region in each image
(the WB-102 body is bright white on a darker desk/background), crops
to its bounding box with a small padding, and rotates to landscape so
the long axis is horizontal.

Intended pipeline position: run BEFORE data/prepare_data.py. Crop the
raw HEICs in data/real/ into data/real_cropped/, then point
prepare_data.py at the cropped directory. This keeps the raw photos
untouched and avoids cropping already-resized 256x256 thumbnails.

Cropped output is always written as PNG (HEIC is read-only via PIL +
pillow-heif). Output filenames keep the source stem with .png suffix.

Usage:
    # Default — crop raw HEICs in data/real to data/real_cropped.
    python scripts/crop_breadboards.py

    # Dry run (logs what would change without writing).
    python scripts/crop_breadboards.py --dry-run

    # Override directories.
    python scripts/crop_breadboards.py \\
        --input-dir path/to/photos --output-dir path/to/cropped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp',
                  '.heic', '.heif'}

# Triage buckets — a cropped output's bbox aspect (long/short) determines
# whether it lands in good/ or needs_review/. Range chosen so the WB-102's
# true 3.06:1 sits comfortably inside, with margin for slight angle/skew.
TRIAGE_ASPECT_RANGE = (1.6, 3.5)

# Reject detections whose bounding box covers less than this fraction
# of the image — those are almost certainly not the breadboard
# (a stray label, a glare highlight, etc.).
MIN_AREA_FRACTION = 0.05

# WB-102 physical dimensions are 165.1mm x 54.0mm → ~3.06:1 in landscape.
# Used by the bbox detector to score and filter contour candidates so a
# white-on-white photo (white paper / white desk under the board) doesn't
# get scored as one giant bright blob.
TARGET_ASPECT = 165.1 / 54.0
ASPECT_GATE = (1.8, 5.0)  # acceptable long/short ratio for a candidate bbox


def detect_breadboard_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Return (x, y, w, h) of the breadboard's bounding box, or None if no
    plausible candidate is found.

    Strategy: Canny edges + heavy dilation, then find external contours.
    The breadboard's hole grid produces a very dense edge cluster that
    survives even when the board sits on a white surface (where Otsu
    thresholding fails because board and background are both bright).

    Among contours, prefer those whose bounding-box aspect ratio falls
    near the WB-102's true 3.06:1; pick the largest such candidate. If
    nothing matches the aspect gate, fall back to the contour with the
    best (area × aspect_match²) score.
    """
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # Dilate enough to merge the hole-pattern edges into one connected blob.
    kernel_size = max(5, min(w_img, h_img) // 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = w_img * h_img
    candidates: list[tuple[int, int, int, int]] = []
    fallback_score = 0.0
    fallback_bbox: tuple[int, int, int, int] | None = None

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = w * h
        if bbox_area / img_area < MIN_AREA_FRACTION:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        aspect_match = 1.0 / (1.0 + abs(aspect - TARGET_ASPECT))

        score = (bbox_area / img_area) * (aspect_match ** 2)
        if score > fallback_score:
            fallback_score = score
            fallback_bbox = (x, y, w, h)

        if ASPECT_GATE[0] <= aspect <= ASPECT_GATE[1]:
            candidates.append((x, y, w, h))

    if candidates:
        # Largest bbox among aspect-matching candidates wins.
        return max(candidates, key=lambda b: b[2] * b[3])
    return fallback_bbox


def crop_with_padding(
    image: np.ndarray, bbox: tuple[int, int, int, int], pad_fraction: float
) -> np.ndarray:
    """Crop to bbox plus padding (clamped to image bounds)."""
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    pad = int(round(max(w, h) * pad_fraction))

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)
    return image[y0:y1, x0:x1]


def to_landscape(image: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees clockwise if the image is portrait."""
    h, w = image.shape[:2]
    if h > w:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def load_image_bgr(path: Path) -> np.ndarray | None:
    """
    Load an image as a BGR uint8 numpy array suitable for OpenCV.

    Uses PIL (with pillow-heif registered) so HEIC/HEIF files work —
    cv2.imread() does not support HEIC.
    """
    try:
        with Image.open(path) as im:
            im = im.convert('RGB')
            rgb = np.asarray(im)
    except Exception:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _triage_bucket(image_w: int, image_h: int) -> str:
    """Return 'good' or 'needs_review' based on the output aspect ratio."""
    long_side = max(image_w, image_h)
    short_side = max(1, min(image_w, image_h))
    aspect = long_side / short_side
    if TRIAGE_ASPECT_RANGE[0] <= aspect <= TRIAGE_ASPECT_RANGE[1]:
        return 'good'
    return 'needs_review'


def process_image(
    src_path: Path,
    output_dir: Path,
    pad_fraction: float,
    dry_run: bool,
    triage: bool,
) -> tuple[str, str | None]:
    """
    Crop one image. Returns (log_message, bucket) where bucket is
    'good' / 'needs_review' / None (skipped).
    """
    image = load_image_bgr(src_path)
    if image is None:
        return f"SKIP (unreadable): {src_path.name}", None

    bbox = detect_breadboard_bbox(image)
    if bbox is None:
        return f"SKIP (no breadboard found): {src_path.name}", None

    cropped = crop_with_padding(image, bbox, pad_fraction)
    cropped = to_landscape(cropped)

    in_h, in_w = image.shape[:2]
    out_h, out_w = cropped.shape[:2]
    bucket = _triage_bucket(out_w, out_h)

    dst_dir = output_dir / bucket if triage else output_dir
    dst_path = dst_dir / f"{src_path.stem}.png"

    msg = (f"OK [{bucket}]: {src_path.name}  {in_w}x{in_h} -> "
           f"{out_w}x{out_h}  (bbox {bbox})")

    if not dry_run:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), cropped)
    return msg, bucket


def process_directory(
    input_dir: Path,
    output_dir: Path,
    pad_fraction: float,
    dry_run: bool,
    triage: bool,
) -> dict:
    """Process every supported image in input_dir."""
    if not input_dir.exists():
        print(f"  [skip] directory does not exist: {input_dir}")
        return {"input_dir": str(input_dir),
                "good": 0, "needs_review": 0, "skipped": 0}

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        print(f"  [skip] no images in {input_dir}")
        return {"input_dir": str(input_dir),
                "good": 0, "needs_review": 0, "skipped": 0}

    print(f"\nProcessing {len(files)} images in {input_dir} -> {output_dir}"
          f"  (triage={'on' if triage else 'off'})")
    counts = {'good': 0, 'needs_review': 0, 'skipped': 0}
    for src in files:
        result, bucket = process_image(
            src, output_dir, pad_fraction, dry_run, triage
        )
        if bucket is None:
            counts['skipped'] += 1
        else:
            counts[bucket] += 1
        print(f"  {result}")

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        **counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop breadboard photos to just the board, normalised to landscape.",
    )
    parser.add_argument(
        '--input-dir', type=Path, action='append', default=None,
        help="Input directory of raw photos (repeatable). "
             "Defaults to data/real/.",
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help="Output directory for cropped PNGs. "
             "Defaults to data/real_cropped/. When --input-dir is given "
             "multiple times, all outputs land in this single directory "
             "(filename collisions overwrite).",
    )
    parser.add_argument(
        '--padding', type=float, default=0.02,
        help="Padding around the detected bounding box, as a fraction of the "
             "longer bbox side. Default: 0.02 (2%%).",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Detect and log only — do not write any files.",
    )
    parser.add_argument(
        '--triage', action=argparse.BooleanOptionalAction, default=True,
        help=f"Sort outputs into good/ (aspect in {TRIAGE_ASPECT_RANGE}) and "
             "needs_review/ (everything else) subdirs. Default: on. "
             "Use --no-triage for a flat output dir.",
    )
    args = parser.parse_args()

    input_dirs = args.input_dir or [PROJECT_ROOT / 'data' / 'real']
    default_output = PROJECT_ROOT / 'data' / 'real_cropped'
    output_dir = args.output_dir if args.output_dir is not None else default_output

    if len(input_dirs) > 1:
        print("WARNING: multiple --input-dirs share one --output-dir; "
              "duplicate filenames will overwrite.", file=sys.stderr)

    totals = {"good": 0, "needs_review": 0, "skipped": 0}
    for in_dir in input_dirs:
        stats = process_directory(
            in_dir, output_dir, args.padding, args.dry_run, args.triage,
        )
        for k in totals:
            totals[k] += stats[k]

    print("\n" + "=" * 60)
    print(f"Done. Good: {totals['good']}  Needs review: {totals['needs_review']}  "
          f"Skipped: {totals['skipped']}")
    if args.triage and not args.dry_run:
        print(f"  → {output_dir}/good/  (clean crops, ready to use)")
        print(f"  → {output_dir}/needs_review/  (hand-fix or discard)")
    if args.dry_run:
        print("(dry run — no files written)")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
