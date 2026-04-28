"""
Phase 8 — CycleGAN training hyperparameters.

These values are read by Phase 8c's training driver. Defaults follow the
canonical CycleGAN paper (Zhu et al., 2017) and the widely-used
junyanz/pytorch-CycleGAN-and-pix2pix reference implementation.

Override at training time via the train.py CLI rather than editing this file.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CycleGANConfig:
    # --- Image / data ---
    image_size: int = 256             # Square crop side, in pixels.

    # --- Optimisation ---
    batch_size: int = 1               # Standard for CycleGAN; instance-norm friendly.
    learning_rate: float = 0.0002     # Adam LR for both G and D.
    beta1: float = 0.5                # Adam beta1 (paper default).
    beta2: float = 0.999              # Adam beta2 (paper default).

    # --- Schedule ---
    epochs: int = 200                 # Total epochs.
    epochs_decay_start: int = 100     # Epoch at which LR begins linear decay to 0.

    # --- Loss weights ---
    lambda_cycle: float = 10.0        # Weight on cycle consistency loss.
    lambda_identity: float = 0.5      # Weight on identity mapping loss
                                      # (helps preserve colour palette).

    # --- Architecture ---
    generator: str = 'resnet_9blocks' # ResNet-9 generator (paper preferred for
                                      # higher-resolution / structure preservation).
    discriminator: str = 'patchgan'   # 70x70 PatchGAN.
    norm: str = 'instance'

    # --- Logging / checkpointing ---
    checkpoint_interval: int = 10     # Save weights every N epochs.
    sample_interval: int = 500        # Save sample translations every N iterations.

    # --- Domain directory names (under data/) ---
    domain_a_name: str = 'synthetic'  # Source: rendered images.
    domain_b_name: str = 'real'       # Target: photographs.


# A single immutable instance other modules can import directly.
CONFIG = CycleGANConfig()


if __name__ == '__main__':
    # Print effective config when invoked directly — useful for sanity-checking
    # before launching a long GPU job.
    from dataclasses import asdict
    import json
    print(json.dumps(asdict(CONFIG), indent=2))
