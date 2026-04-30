"""
CycleGAN training loop — Phase 8c.

Trains two generators (G: A→B, F: B→A) and two discriminators (D_B, D_A)
to translate synthetic breadboard renders into photorealistic images and
back, using cycle-consistency to preserve spatial structure.

Run from the project root:

    python -m cyclegan.train
    python -m cyclegan.train --resume cyclegan/checkpoints/epoch_080.pth
    python -m cyclegan.train --data-root /scratch/breadboard/data \
                             --checkpoints-dir /scratch/breadboard/ckpt \
                             --samples-dir   /scratch/breadboard/samples
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F_nn
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cyclegan.config import CONFIG, CycleGANConfig
from cyclegan.dataset import UnpairedImageDataset, denormalize
from cyclegan.models import build_models


# ---------------------------------------------------------------------------
# Replay buffer (Shrivastava et al. 2017, used in CycleGAN)
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """
    Stores up to `max_size` previously-generated images. When queried with
    a new batch, returns a mix of new and historical samples to stabilise
    discriminator training.

    Per-item rule (from the original paper):
      - With prob 0.5, return the new image and store it.
      - With prob 0.5, return a random historical image and replace it
        with the new one in the buffer.
    """

    def __init__(self, max_size: int = 50):
        if max_size <= 0:
            raise ValueError("ReplayBuffer max_size must be > 0")
        self.max_size = max_size
        self.buffer: list[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        out = []
        for img in images:
            img = img.detach().unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                out.append(img)
            else:
                if random.random() < 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    swapped = self.buffer[idx].clone()
                    self.buffer[idx] = img
                    out.append(swapped)
                else:
                    out.append(img)
        return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Learning-rate scheduler (linear decay to zero after warm period)
# ---------------------------------------------------------------------------


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    decay_start_epoch: int,
) -> optim.lr_scheduler.LambdaLR:
    """LR is held flat for `decay_start_epoch` epochs, then decays linearly to 0."""

    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start_epoch:
            return 1.0
        progress = (epoch - decay_start_epoch) / max(1, total_epochs - decay_start_epoch)
        return max(0.0, 1.0 - progress)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def gan_loss(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """LSGAN loss: MSE against a tensor of all-ones / all-zeros."""
    target_val = 1.0 if target_is_real else 0.0
    target = torch.full_like(pred, target_val)
    return F_nn.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    epoch: int,
    G: nn.Module, F: nn.Module, D_A: nn.Module, D_B: nn.Module,
    opt_G: optim.Optimizer, opt_D: optim.Optimizer,
    sched_G: optim.lr_scheduler._LRScheduler,
    sched_D: optim.lr_scheduler._LRScheduler,
    config: CycleGANConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'G': G.state_dict(),
        'F': F.state_dict(),
        'D_A': D_A.state_dict(),
        'D_B': D_B.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
        'sched_G': sched_G.state_dict(),
        'sched_D': sched_D.state_dict(),
        'config': asdict(config),
    }, path)


def load_checkpoint(
    path: Path,
    G: nn.Module, F: nn.Module, D_A: nn.Module, D_B: nn.Module,
    opt_G: optim.Optimizer, opt_D: optim.Optimizer,
    sched_G: optim.lr_scheduler._LRScheduler,
    sched_D: optim.lr_scheduler._LRScheduler,
    map_location: str | torch.device,
) -> int:
    """Restore all networks/optimizers/schedulers in-place. Returns next epoch index."""
    state = torch.load(path, map_location=map_location)
    G.load_state_dict(state['G'])
    F.load_state_dict(state['F'])
    D_A.load_state_dict(state['D_A'])
    D_B.load_state_dict(state['D_B'])
    opt_G.load_state_dict(state['opt_G'])
    opt_D.load_state_dict(state['opt_D'])
    sched_G.load_state_dict(state['sched_G'])
    sched_D.load_state_dict(state['sched_D'])
    return int(state['epoch']) + 1


# ---------------------------------------------------------------------------
# Sample export
# ---------------------------------------------------------------------------


def save_samples(
    epoch: int,
    G: nn.Module, F: nn.Module,
    fixed_a: torch.Tensor, fixed_b: torch.Tensor,
    samples_dir: Path,
) -> None:
    """Save a side-by-side grid: real_A | G(A) | F(G(A)) | real_B | F(B) | G(F(B))."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    G.eval(); F.eval()
    with torch.no_grad():
        fake_b = G(fixed_a)
        rec_a = F(fake_b)
        fake_a = F(fixed_b)
        rec_b = G(fake_a)

    grid = torch.cat([fixed_a, fake_b, rec_a, fixed_b, fake_a, rec_b], dim=0)
    grid = denormalize(grid)
    out_path = samples_dir / f"epoch_{epoch:04d}.png"
    save_image(grid, out_path, nrow=fixed_a.size(0))
    G.train(); F.train()


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------


CSV_FIELDS = [
    'epoch', 'iter', 'global_step',
    'loss_G', 'loss_G_gan_AB', 'loss_G_gan_BA',
    'loss_cycle_A', 'loss_cycle_B',
    'loss_idt_A', 'loss_idt_B',
    'loss_D_A', 'loss_D_B',
    'lr_G', 'lr_D',
    'wallclock_s',
]


class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        self._fh = open(self.path, 'a', newline='')
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    config: CycleGANConfig,
    data_root: Path,
    checkpoints_dir: Path,
    samples_dir: Path,
    log_path: Path,
    resume_from: Path | None,
    device_str: str | None,
    num_workers: int,
    seed: int,
) -> None:
    # Reproducibility (best-effort — true determinism on GPU also needs CUDNN flags).
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        device_str if device_str is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    if device.type == 'cpu':
        print("WARNING: training on CPU — this will be very slow. "
              "CycleGAN expects a GPU.")

    # --- Data ---
    train_root_a = data_root / config.domain_a_name / 'train'
    train_root_b = data_root / config.domain_b_name / 'train'
    test_root_a = data_root / config.domain_a_name / 'test'
    test_root_b = data_root / config.domain_b_name / 'test'

    train_ds = UnpairedImageDataset(
        train_root_a, train_root_b, image_size=config.image_size, train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"Train: {len(train_ds.files_a)} A images, {len(train_ds.files_b)} B images, "
          f"{len(train_ds)} pairs/epoch")

    # Fixed sample batch for monitoring (drawn once from the test split if
    # available, else from train).
    sample_root_a = test_root_a if test_root_a.exists() and any(test_root_a.iterdir()) else train_root_a
    sample_root_b = test_root_b if test_root_b.exists() and any(test_root_b.iterdir()) else train_root_b
    sample_ds = UnpairedImageDataset(
        sample_root_a, sample_root_b, image_size=config.image_size, train=False,
    )
    n_samples = min(4, len(sample_ds))
    fixed_a = torch.stack([sample_ds[i]['A'] for i in range(n_samples)]).to(device)
    fixed_b = torch.stack([sample_ds[i]['B'] for i in range(n_samples)]).to(device)

    # --- Models ---
    G, F, D_B, D_A = build_models(image_channels=3, n_residual_blocks=9)
    G.to(device); F.to(device); D_A.to(device); D_B.to(device)

    # --- Optimizers ---
    opt_G = optim.Adam(
        itertools.chain(G.parameters(), F.parameters()),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    opt_D = optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    sched_G = build_lr_scheduler(opt_G, config.epochs, config.epochs_decay_start)
    sched_D = build_lr_scheduler(opt_D, config.epochs, config.epochs_decay_start)

    start_epoch = 0
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        start_epoch = load_checkpoint(
            resume_from, G, F, D_A, D_B, opt_G, opt_D, sched_G, sched_D,
            map_location=device,
        )
        print(f"  resumed at epoch {start_epoch}")

    # Replay buffers, one per discriminator.
    buffer_fake_A = ReplayBuffer(max_size=50)
    buffer_fake_B = ReplayBuffer(max_size=50)

    logger = CSVLogger(log_path)
    t0 = time.time()
    global_step = 0

    try:
        for epoch in range(start_epoch, config.epochs):
            G.train(); F.train(); D_A.train(); D_B.train()

            for it, batch in enumerate(train_loader):
                real_A = batch['A'].to(device, non_blocking=True)
                real_B = batch['B'].to(device, non_blocking=True)

                # ============ Generators G + F ============
                opt_G.zero_grad(set_to_none=True)

                # Identity: G(real_B) should == real_B (preserves colour);
                # F(real_A) should == real_A.
                if config.lambda_identity > 0:
                    idt_B = G(real_B)
                    loss_idt_B = F_nn.l1_loss(idt_B, real_B) * config.lambda_cycle * config.lambda_identity
                    idt_A = F(real_A)
                    loss_idt_A = F_nn.l1_loss(idt_A, real_A) * config.lambda_cycle * config.lambda_identity
                else:
                    loss_idt_A = torch.tensor(0.0, device=device)
                    loss_idt_B = torch.tensor(0.0, device=device)

                # GAN losses for the generators (try to fool the discriminators).
                fake_B = G(real_A)
                loss_G_gan_AB = gan_loss(D_B(fake_B), target_is_real=True)
                fake_A = F(real_B)
                loss_G_gan_BA = gan_loss(D_A(fake_A), target_is_real=True)

                # Cycle consistency.
                rec_A = F(fake_B)
                loss_cycle_A = F_nn.l1_loss(rec_A, real_A) * config.lambda_cycle
                rec_B = G(fake_A)
                loss_cycle_B = F_nn.l1_loss(rec_B, real_B) * config.lambda_cycle

                loss_G = (
                    loss_G_gan_AB + loss_G_gan_BA
                    + loss_cycle_A + loss_cycle_B
                    + loss_idt_A + loss_idt_B
                )
                loss_G.backward()
                opt_G.step()

                # ============ Discriminator D_B (judges B-domain) ============
                opt_D.zero_grad(set_to_none=True)

                fake_B_buffered = buffer_fake_B.push_and_pop(fake_B)
                pred_real_B = D_B(real_B)
                pred_fake_B = D_B(fake_B_buffered.detach())
                loss_D_B = 0.5 * (
                    gan_loss(pred_real_B, target_is_real=True)
                    + gan_loss(pred_fake_B, target_is_real=False)
                )
                loss_D_B.backward()

                # ============ Discriminator D_A (judges A-domain) ============
                fake_A_buffered = buffer_fake_A.push_and_pop(fake_A)
                pred_real_A = D_A(real_A)
                pred_fake_A = D_A(fake_A_buffered.detach())
                loss_D_A = 0.5 * (
                    gan_loss(pred_real_A, target_is_real=True)
                    + gan_loss(pred_fake_A, target_is_real=False)
                )
                loss_D_A.backward()
                opt_D.step()

                global_step += 1

                # Console + CSV logging.
                if global_step % 50 == 0 or it == 0:
                    lr_g = opt_G.param_groups[0]['lr']
                    lr_d = opt_D.param_groups[0]['lr']
                    elapsed = time.time() - t0
                    print(
                        f"[ep {epoch+1}/{config.epochs}] "
                        f"it {it+1}/{len(train_loader)}  "
                        f"G {loss_G.item():.3f} "
                        f"(gan {loss_G_gan_AB.item():.2f}/{loss_G_gan_BA.item():.2f}, "
                        f"cyc {loss_cycle_A.item():.2f}/{loss_cycle_B.item():.2f}, "
                        f"idt {loss_idt_A.item():.2f}/{loss_idt_B.item():.2f})  "
                        f"D_A {loss_D_A.item():.3f}  D_B {loss_D_B.item():.3f}  "
                        f"lr {lr_g:.2e}  t {elapsed:.0f}s"
                    )
                    logger.log({
                        'epoch': epoch + 1,
                        'iter': it + 1,
                        'global_step': global_step,
                        'loss_G': loss_G.item(),
                        'loss_G_gan_AB': loss_G_gan_AB.item(),
                        'loss_G_gan_BA': loss_G_gan_BA.item(),
                        'loss_cycle_A': loss_cycle_A.item(),
                        'loss_cycle_B': loss_cycle_B.item(),
                        'loss_idt_A': loss_idt_A.item(),
                        'loss_idt_B': loss_idt_B.item(),
                        'loss_D_A': loss_D_A.item(),
                        'loss_D_B': loss_D_B.item(),
                        'lr_G': lr_g,
                        'lr_D': lr_d,
                        'wallclock_s': round(elapsed, 1),
                    })

            # End of epoch.
            sched_G.step()
            sched_D.step()

            # Save sample translations every 5 epochs (and at epoch 1).
            if (epoch + 1) % 5 == 0 or epoch == 0:
                save_samples(epoch + 1, G, F, fixed_a, fixed_b, samples_dir)

            # Save checkpoint every checkpoint_interval epochs and at the end.
            if (epoch + 1) % config.checkpoint_interval == 0 or (epoch + 1) == config.epochs:
                ckpt_path = checkpoints_dir / f"epoch_{epoch+1:03d}.pth"
                save_checkpoint(
                    ckpt_path, epoch, G, F, D_A, D_B,
                    opt_G, opt_D, sched_G, sched_D, config,
                )
                print(f"  -> saved checkpoint {ckpt_path}")

    finally:
        logger.close()

    print(f"Training complete in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CycleGAN (synthetic ↔ real).")
    parser.add_argument(
        '--data-root', type=Path,
        default=Path(__file__).resolve().parent.parent / 'data',
        help="Top-level data directory containing synthetic/ and real/ subfolders.",
    )
    parser.add_argument(
        '--checkpoints-dir', type=Path,
        default=Path(__file__).resolve().parent / 'checkpoints',
    )
    parser.add_argument(
        '--samples-dir', type=Path,
        default=Path(__file__).resolve().parent / 'samples',
    )
    parser.add_argument(
        '--log-path', type=Path,
        default=Path(__file__).resolve().parent / 'logs' / 'train_loss.csv',
    )
    parser.add_argument('--resume', type=Path, default=None,
                        help="Path to checkpoint .pth to resume from.")
    parser.add_argument('--device', type=str, default=None,
                        help="Override device (e.g. 'cuda:0' or 'cpu').")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Optional overrides for the most common hyperparameters. When omitted,
    # the corresponding default from cyclegan.config.CONFIG is used.
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override total epochs (default from config.py).")
    parser.add_argument('--epochs-decay-start', type=int, default=None,
                        help="Override epoch at which LR begins linear decay "
                             "to zero (default from config.py).")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Override batch size (default from config.py).")
    parser.add_argument('--lr', type=float, default=None,
                        help="Override Adam learning rate.")
    parser.add_argument('--image-size', type=int, default=None,
                        help="Square training resolution in pixels. Overrides "
                             "config.image_size (default 256). Pass 512 to "
                             "train at 512x512 — make sure prepare_data.py "
                             "was run with the same --image-size.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = CONFIG
    overrides: dict = {}
    if args.epochs is not None: overrides['epochs'] = args.epochs
    if args.epochs_decay_start is not None: overrides['epochs_decay_start'] = args.epochs_decay_start
    if args.batch_size is not None: overrides['batch_size'] = args.batch_size
    if args.lr is not None: overrides['learning_rate'] = args.lr
    if args.image_size is not None: overrides['image_size'] = args.image_size
    if overrides:
        cfg = replace(cfg, **overrides)

    print("Effective config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    train(
        config=cfg,
        data_root=args.data_root,
        checkpoints_dir=args.checkpoints_dir,
        samples_dir=args.samples_dir,
        log_path=args.log_path,
        resume_from=args.resume,
        device_str=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
