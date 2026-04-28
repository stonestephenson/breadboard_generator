"""
CycleGAN model architectures — implemented directly from Zhu et al. (2017),
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks".

Two networks are exposed:

  ResnetGenerator    — 9-block ResNet generator used by the paper for 256x256
                       images. Mapping function for one direction of the cycle.

  PatchGANDiscriminator — 70x70 PatchGAN classifier. Outputs a feature map
                       where each location predicts whether its receptive
                       field is real or fake.

Both use InstanceNorm (per the paper) and reflection padding on the outer
7x7 convolutions to suppress edge artefacts.
"""

from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ResnetBlock(nn.Module):
    """Residual block: two 3x3 conv-InstanceNorm-ReLU layers with reflection pad."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    Paper architecture (256x256 input):

        c7s1-64, d128, d256, R256 x 9, u128, u64, c7s1-3

    where:
        c7s1-k : 7x7 Conv-InstanceNorm-ReLU, k filters, stride 1, ReflectionPad
        dk     : 3x3 Conv-InstanceNorm-ReLU, k filters, stride 2
        Rk     : ResnetBlock with k channels
        uk     : 3x3 ConvTranspose-InstanceNorm-ReLU, k filters, stride 2
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        n_residual_blocks: int = 9,
    ):
        super().__init__()

        layers: list[nn.Module] = [
            # c7s1-64
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        ]

        # Downsampling: d128, d256
        ch = base_filters
        for _ in range(2):
            layers += [
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        # Residual stack
        for _ in range(n_residual_blocks):
            layers.append(ResnetBlock(ch))

        # Upsampling: u128, u64
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    ch, ch // 2, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=True,
                ),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2

        # c7s1-3 with tanh to map back to [-1, 1]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Discriminator (70x70 PatchGAN)
# ---------------------------------------------------------------------------


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN: C64-C128-C256-C512, then a final 1-channel conv.

    Ck is a 4x4 conv-InstanceNorm-LeakyReLU(0.2), stride 2 except the last
    Ck which uses stride 1 (paper Section 7.2). C64 has no InstanceNorm.

    Output is a single-channel feature map (no sigmoid — losses operate on
    logits/raw values; LSGAN uses MSE directly on the output).
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 64):
        super().__init__()

        def block(in_c: int, out_c: int, stride: int, norm: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not norm),
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers: list[nn.Module] = []
        # C64 — no norm
        layers += block(in_channels, base_filters, stride=2, norm=False)
        # C128
        layers += block(base_filters, base_filters * 2, stride=2, norm=True)
        # C256
        layers += block(base_filters * 2, base_filters * 4, stride=2, norm=True)
        # C512 with stride 1 (final feature block before classifier)
        layers += block(base_filters * 4, base_filters * 8, stride=1, norm=True)
        # 1-channel patch output, stride 1
        layers.append(
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Weight init (paper: N(0, 0.02))
# ---------------------------------------------------------------------------


def init_weights(module: nn.Module, gain: float = 0.02) -> None:
    """Apply N(0, gain) init to Conv / ConvTranspose weights, ones to norm scale."""
    classname = module.__class__.__name__
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif 'InstanceNorm' in classname or 'BatchNorm' in classname:
        if getattr(module, 'weight', None) is not None:
            nn.init.normal_(module.weight.data, 1.0, gain)
        if getattr(module, 'bias', None) is not None:
            nn.init.zeros_(module.bias.data)


def build_models(
    image_channels: int = 3,
    n_residual_blocks: int = 9,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Construct the four CycleGAN networks with paper-default sizes and inits.

    Returns:
        (G, F, D_B, D_A)
        G   — synthetic (A) → real     (B)
        F   — real      (B) → synthetic (A)
        D_B — judges images of domain B (real)
        D_A — judges images of domain A (synthetic)
    """
    G = ResnetGenerator(image_channels, image_channels, n_residual_blocks=n_residual_blocks)
    F = ResnetGenerator(image_channels, image_channels, n_residual_blocks=n_residual_blocks)
    D_B = PatchGANDiscriminator(image_channels)
    D_A = PatchGANDiscriminator(image_channels)
    for net in (G, F, D_B, D_A):
        net.apply(init_weights)
    return G, F, D_B, D_A
