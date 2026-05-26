import os
import random
from io import BytesIO
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
# Issue #15: removed duplicate `import torchvision.transforms.functional as F`

class RandomJPEGCompression:
    """
    Simulate JPEG compression artifacts.
    """
    def __init__(self, quality_range=(30, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img

        buffer = BytesIO()
        quality = random.randint(*self.quality_range)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

class RandomResizeJitter:
    """
    Resize to random scale then back to target size.
    """
    def __init__(self, scale=(0.8, 1.2), size=256):
        self.scale = scale
        self.size = size

    def __call__(self, img: Image.Image):
        w, h = img.size
        scale = random.uniform(*self.scale)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        img = TF.resize(img, (new_h, new_w))

        # Issue #8: guard crop size against images smaller than requested crop
        crop_size = min(new_w, new_h, self.size)
        img = TF.center_crop(img, crop_size)
        img = TF.resize(img, (self.size, self.size))
        return img

class DeepfakeDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int, int]],
        image_size: int = 256,
        augment: bool = True,
        aug_grayscale_p: float = 0.2,
        aug_color_jitter_p: float = 0.3,
    ):
        """
        Args:
            samples: list of (image_path, label, generator_id)
            image_size: target dimension for resizing
            augment: whether to apply random augmentations
        """
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

        # Removed _jpeg_cache: When num_workers > 0, each worker gets a copy of the
        # cache which leads to massive memory leaks (OOM). We encode on the fly instead.

        # Base transform
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
        ])

        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomGrayscale(p=aug_grayscale_p),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=aug_color_jitter_p),
            RandomJPEGCompression(p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        ])

        # Final tensor conversion
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def _standardize_image(self, idx, img: Image.Image):
        """Force non-JPEG source formats through a JPEG path to reduce
        PNG-vs-JPEG and source-compression shortcuts without bottlenecking I/O.
        """
        img_path = self.samples[idx][0]
        _, ext = os.path.splitext(img_path)
        
        # If the original file was a JPEG, we can skip the standardization
        if ext.lower() in (".jpg", ".jpeg", ".mpo"):
            return img
            
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, gen_id = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Resize first (ensures consistent base)
        img = self.base_transform(img)

        # Standardize before augmentation so train/val/test share the same source-compression baseline.
        img = self._standardize_image(idx, img)

        # Apply augmentations
        if self.augment:
            img = self.augment_transform(img)

        # To tensor
        img = self.to_tensor(img)

        # Extract video identity — must include gen_id to disambiguate.
        # In FF++, real/000_frame_0001.jpg and fake/Deepfakes/000_frame_0001.jpg
        # share the same base video ID "000". Without gen_id prefix, the
        # video-level aggregation in evaluate() merges real and fake frames
        # under the same video, scrambling labels and producing AUC < 0.5.
        basename = os.path.basename(img_path)
        if "_frame" in basename:
            base_vid = basename.split("_frame")[0]
        else:
            base_vid = os.path.splitext(basename)[0]
            
        video_id = f"{gen_id}_{base_vid}"

        result = {
            "image":  img,                                         # (3, H, W)
            "label":  torch.tensor(label,  dtype=torch.long),
            "gen_name": gen_id,
            "video_id": video_id,
        }

        return result
