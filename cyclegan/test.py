"""
CycleGAN inference — Phase 8c.

Loads a trained generator G (synthetic → real) from a training checkpoint and
applies it to every image in an input directory.

    python -m cyclegan.test \
        --checkpoint cyclegan/checkpoints/epoch_200.pth \
        --input data/synthetic/test/ \
        --output output/stylized/

The output directory will contain one PNG per input, with the same stem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cyclegan.dataset import SingleDomainDataset, denormalize
from cyclegan.models import ResnetGenerator


def load_generator(
    checkpoint_path: Path,
    device: torch.device,
    direction: str = 'G',
) -> ResnetGenerator:
    """
    Load just the generator weights from a training checkpoint.

    direction: 'G' for synthetic→real (default), 'F' for real→synthetic.
    """
    if direction not in ('G', 'F'):
        raise ValueError(f"direction must be 'G' or 'F', got {direction!r}")

    state = torch.load(checkpoint_path, map_location=device)

    # Pull generator hyperparameters from the embedded config when present.
    cfg = state.get('config', {}) or {}
    n_blocks = 9
    image_channels = 3
    if cfg.get('generator', '').startswith('resnet_'):
        try:
            n_blocks = int(cfg['generator'].split('_')[1].rstrip('blocks'))
        except (IndexError, ValueError):
            n_blocks = 9

    net = ResnetGenerator(
        in_channels=image_channels,
        out_channels=image_channels,
        n_residual_blocks=n_blocks,
    )
    net.load_state_dict(state[direction])
    net.to(device)
    net.eval()
    return net


def translate_directory(
    checkpoint_path: Path,
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    direction: str,
    device_str: str | None,
    batch_size: int,
    num_workers: int,
) -> int:
    device = torch.device(
        device_str if device_str is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    net = load_generator(checkpoint_path, device, direction=direction)
    ds = SingleDomainDataset(input_dir, image_size=image_size)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
    )

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            translated = net(imgs)
            translated = denormalize(translated)
            for i, name in enumerate(batch['name']):
                out_path = output_dir / f"{name}.png"
                save_image(translated[i], out_path)
                n_done += 1
            if n_done % 50 == 0:
                print(f"  translated {n_done}/{len(ds)}")

    print(f"Done. {n_done} images written to {output_dir}")
    return n_done


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate images with a trained CycleGAN generator.")
    p.add_argument('--checkpoint', type=Path, required=True,
                   help="Path to training checkpoint .pth.")
    p.add_argument('--input', type=Path, required=True,
                   help="Directory of source-domain images.")
    p.add_argument('--output', type=Path, required=True,
                   help="Directory for translated outputs.")
    p.add_argument('--direction', choices=('G', 'F'), default='G',
                   help="'G' = synthetic→real (default), 'F' = real→synthetic.")
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    translate_directory(
        checkpoint_path=args.checkpoint,
        input_dir=args.input,
        output_dir=args.output,
        image_size=args.image_size,
        direction=args.direction,
        device_str=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
