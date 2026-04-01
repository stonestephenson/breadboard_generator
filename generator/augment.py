"""
Augmentation pipeline — OpenCV post-processing for domain gap reduction.

Applies realistic distortions to rendered breadboard images so synthetic
training data looks closer to real photos. All augmentations are seeded
and parameterized via augmentation_config.json.
"""

import json
import random
import numpy as np
import cv2
from PIL import Image


def load_augmentation_config(config_path: str) -> dict:
    """Load augmentation config from JSON file."""
    with open(config_path) as f:
        return json.load(f)


class AugmentationPipeline:
    """Applies parameterized augmentations to breadboard images."""

    def __init__(self, config: dict, seed: int = 42):
        """
        Args:
            config: Augmentation config dict (from augmentation_config.json).
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def _to_cv(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Convert PIL Image to OpenCV BGR numpy array."""
        if isinstance(img, Image.Image):
            arr = np.array(img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return img

    def _to_pil(self, arr: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR array to PIL RGB Image."""
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def perspective_warp(
        self, img: Image.Image | np.ndarray, max_angle_deg: float | None = None,
    ) -> np.ndarray:
        """
        Simulate camera not being perfectly overhead.

        Applies a random perspective transform by displacing corners.
        """
        arr = self._to_cv(img)
        h, w = arr.shape[:2]

        if max_angle_deg is None:
            max_angle_deg = self.config['perspective_warp']['max_angle_deg']

        # Convert angle to pixel displacement (approximate)
        max_disp = int(w * np.tan(np.radians(max_angle_deg)) * 0.1)
        max_disp = max(1, max_disp)

        def _rand_disp():
            return self.rng.randint(-max_disp, max_disp)

        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [_rand_disp(), _rand_disp()],
            [w + _rand_disp(), _rand_disp()],
            [w + _rand_disp(), h + _rand_disp()],
            [_rand_disp(), h + _rand_disp()],
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return result

    def lighting_jitter(
        self, img: Image.Image | np.ndarray,
        brightness_range: tuple[int, int] | None = None,
        contrast_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Random brightness and contrast shifts."""
        arr = self._to_cv(img).astype(np.float32)

        cfg = self.config['lighting_jitter']
        if brightness_range is None:
            brightness_range = tuple(cfg['brightness_range'])
        if contrast_range is None:
            contrast_range = tuple(cfg['contrast_range'])

        brightness = self.rng.uniform(brightness_range[0], brightness_range[1])
        contrast = self.rng.uniform(contrast_range[0], contrast_range[1])

        arr = arr * contrast + brightness
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def gaussian_blur(
        self, img: Image.Image | np.ndarray, kernel_range: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Simulate slight out-of-focus."""
        arr = self._to_cv(img)

        if kernel_range is None:
            kernel_range = tuple(self.config['gaussian_blur']['kernel_range'])

        k = self.rng.randint(kernel_range[0], kernel_range[1])
        # Kernel must be odd
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(arr, (k, k), 0)

    def rotation(
        self, img: Image.Image | np.ndarray, max_angle_deg: float | None = None,
    ) -> np.ndarray:
        """Slight in-plane rotation."""
        arr = self._to_cv(img)
        h, w = arr.shape[:2]

        if max_angle_deg is None:
            max_angle_deg = self.config['rotation']['max_angle_deg']

        angle = self.rng.uniform(-max_angle_deg, max_angle_deg)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def add_shadow(
        self, img: Image.Image | np.ndarray,
        shadow_intensity_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Overlay a random gradient shadow across the image."""
        arr = self._to_cv(img).astype(np.float32)
        h, w = arr.shape[:2]

        if shadow_intensity_range is None:
            shadow_intensity_range = tuple(self.config['add_shadow']['shadow_intensity_range'])

        intensity = self.rng.uniform(shadow_intensity_range[0], shadow_intensity_range[1])

        # Random gradient direction
        direction = self.rng.choice(['horizontal', 'vertical', 'diagonal'])

        if direction == 'horizontal':
            gradient = np.linspace(1.0, 1.0 - intensity, w, dtype=np.float32)
            shadow = np.tile(gradient, (h, 1))
        elif direction == 'vertical':
            gradient = np.linspace(1.0, 1.0 - intensity, h, dtype=np.float32)
            shadow = np.tile(gradient.reshape(-1, 1), (1, w))
        else:
            gx = np.linspace(0, 1, w, dtype=np.float32)
            gy = np.linspace(0, 1, h, dtype=np.float32)
            mx, my = np.meshgrid(gx, gy)
            shadow = 1.0 - intensity * (mx + my) / 2.0

        # Randomly flip the gradient direction
        if self.rng.random() > 0.5:
            shadow = np.flip(shadow, axis=1).copy()
        if self.rng.random() > 0.5:
            shadow = np.flip(shadow, axis=0).copy()

        shadow = shadow[:, :, np.newaxis]
        arr = arr * shadow
        return np.clip(arr, 0, 255).astype(np.uint8)

    def add_noise(
        self, img: Image.Image | np.ndarray,
        noise_std_range: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Gaussian sensor noise."""
        arr = self._to_cv(img).astype(np.float32)

        if noise_std_range is None:
            noise_std_range = tuple(self.config['add_noise']['noise_std_range'])

        std = self.rng.uniform(noise_std_range[0], noise_std_range[1])
        noise = self.np_rng.normal(0, std, arr.shape).astype(np.float32)
        arr = arr + noise
        return np.clip(arr, 0, 255).astype(np.uint8)

    def background_variation(
        self, img: Image.Image | np.ndarray, bg_colors: list | None = None,
    ) -> np.ndarray:
        """
        Replace the background color around the board with a different surface color.

        Detects near-grey background pixels and replaces them.
        """
        arr = self._to_cv(img)

        if bg_colors is None:
            bg_colors = self.config['background_variation']['bg_colors']

        new_bg = tuple(self.rng.choice(bg_colors))
        # Convert RGB config to BGR for OpenCV
        new_bg_bgr = (new_bg[2], new_bg[1], new_bg[0])

        # Detect background: pixels close to the original grey (180, 180, 180)
        grey_bgr = np.array([180, 180, 180], dtype=np.uint8)
        diff = np.abs(arr.astype(np.int16) - grey_bgr.astype(np.int16))
        mask = np.all(diff < 30, axis=2)

        arr[mask] = new_bg_bgr
        return arr

    def apply_random(
        self, img: Image.Image | np.ndarray,
        n_augmentations: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Apply a random subset of enabled augmentations.

        Args:
            img: Input image.
            n_augmentations: (min, max) number of augmentations to apply.

        Returns:
            (augmented_image_bgr, augmentation_record)
        """
        if n_augmentations is None:
            n_augmentations = tuple(self.config['apply_random']['n_augmentations'])

        available = []
        aug_methods = {
            'perspective_warp': self.perspective_warp,
            'lighting_jitter': self.lighting_jitter,
            'gaussian_blur': self.gaussian_blur,
            'rotation': self.rotation,
            'add_shadow': self.add_shadow,
            'add_noise': self.add_noise,
            'background_variation': self.background_variation,
        }

        for name, method in aug_methods.items():
            if self.config.get(name, {}).get('enabled', True):
                available.append((name, method))

        n = self.rng.randint(n_augmentations[0], n_augmentations[1])
        n = min(n, len(available))
        selected = self.rng.sample(available, n)

        arr = self._to_cv(img)
        applied = []

        for name, method in selected:
            arr = method(arr)
            applied.append(name)

        record = {
            "augmentations": applied,
            "n_applied": len(applied),
            "seed": self.seed,
        }
        return arr, record

    def apply_random_pil(
        self, img: Image.Image,
        n_augmentations: tuple[int, int] | None = None,
    ) -> tuple[Image.Image, dict]:
        """Convenience wrapper that returns a PIL Image."""
        arr, record = self.apply_random(img, n_augmentations)
        return self._to_pil(arr), record
