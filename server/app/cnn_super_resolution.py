"""
CNN/Transformer-based Super-Resolution for Satellite Imagery.

Supports:
- Real-ESRGAN (GAN-based, best for photo-realism)
- RRDB Network (CNN-based, stable)

These provide TRUE AI super-resolution that hallucinates realistic detail.
"""

import os
import urllib.request
from pathlib import Path
from typing import Tuple, Optional
import json
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model configurations
MODELS = {
    "realesrgan_x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "channels": 64,
        "blocks": 23,
    },
    "realesrgan_x2": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "channels": 64,
        "blocks": 23,
    },
}


def get_model_dir() -> Path:
    """Get model directory."""
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir


def download_weights(model_name: str) -> Path:
    """Download model weights if not present."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODELS[model_name]
    model_dir = get_model_dir()
    weights_path = model_dir / f"{model_name}.pth"

    if not weights_path.exists():
        print(f"📥 Downloading {model_name} weights...")
        urllib.request.urlretrieve(config["url"], weights_path)
        size_mb = weights_path.stat().st_size / 1024 / 1024
        print(f"   ✅ Downloaded ({size_mb:.1f} MB)")

    return weights_path


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDB."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDB Network for Real-ESRGAN."""

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers - both defined but conditionally used
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # First upsample (always done for both x2 and x4)
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        
        # Second upsample (only for x4)
        if self.scale == 4:
            feat = self.lrelu(
                self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )

        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        return out


class RealESRGAN:
    """Real-ESRGAN inference wrapper."""

    def __init__(self, scale: int = 4, device: str = None, tile_size: int = 256):
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = 10

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"   Device: {self.device}")

        model_name = f"realesrgan_x{scale}"
        weights_path = download_weights(model_name)

        config = MODELS[model_name]
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=config["channels"],
            num_block=config["blocks"],
            num_grow_ch=32,
            scale=scale,
        )

        state_dict = torch.load(weights_path, map_location=self.device)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

        print(f"   ✅ Loaded Real-ESRGAN x{scale}")

    @torch.no_grad()
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """Enhance image using Real-ESRGAN."""
        img_t = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_t).permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.to(self.device)

        h, w = img_t.shape[2:]

        if h * w > self.tile_size * self.tile_size * 4:
            output = self._tile_process(img_t)
        else:
            output = self.model(img_t)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        return output

    def _tile_process(self, img: torch.Tensor) -> torch.Tensor:
        """Process image in tiles to save memory."""
        batch, channel, height, width = img.shape
        output_h = height * self.scale
        output_w = width * self.scale

        output = torch.zeros((batch, channel, output_h, output_w), device=self.device)

        tiles_x = (width + self.tile_size - 1) // self.tile_size
        tiles_y = (height + self.tile_size - 1) // self.tile_size

        for y in range(tiles_y):
            for x in range(tiles_x):
                x1 = x * self.tile_size
                y1 = y * self.tile_size
                x2 = min(x1 + self.tile_size + self.tile_pad * 2, width)
                y2 = min(y1 + self.tile_size + self.tile_pad * 2, height)
                x1 = max(x2 - self.tile_size - self.tile_pad * 2, 0)
                y1 = max(y2 - self.tile_size - self.tile_pad * 2, 0)

                tile = img[:, :, y1:y2, x1:x2]
                tile_out = self.model(tile)

                out_x1 = x1 * self.scale
                out_y1 = y1 * self.scale
                out_x2 = x2 * self.scale
                out_y2 = y2 * self.scale

                pad = self.tile_pad * self.scale
                if x > 0:
                    tile_out = tile_out[:, :, :, pad:]
                    out_x1 += pad
                if y > 0:
                    tile_out = tile_out[:, :, pad:, :]
                    out_y1 += pad
                if x < tiles_x - 1:
                    tile_out = tile_out[:, :, :, :-pad]
                    out_x2 -= pad
                if y < tiles_y - 1:
                    tile_out = tile_out[:, :, :-pad, :]
                    out_y2 -= pad

                output[:, :, out_y1:out_y2, out_x1:out_x2] = tile_out

        return output


def apply_cnn_sr(
    input_path: Path,
    output_path: Path,
    scale: int = 4,
) -> Tuple[Path, dict]:
    """Apply CNN-based super-resolution (Real-ESRGAN)."""
    import rasterio
    from rasterio.transform import Affine

    print(f"\n🔬 CNN Super-Resolution (Real-ESRGAN x{scale})")
    print(f"   Input: {input_path}")

    input_path = Path(input_path)
    transform = None
    crs = None

    if input_path.suffix.lower() in [".tif", ".tiff"]:
        with rasterio.open(input_path) as src:
            if src.count >= 3:
                img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
            else:
                img = src.read(1)
                img = np.stack([img, img, img], axis=-1)

            if img.dtype != np.uint8:
                if img.max() > 255:
                    img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
                img = img.astype(np.uint8)

            transform = src.transform
            crs = src.crs
    else:
        img = cv2.imread(str(input_path))

    print(f"   Input size: {img.shape[1]}x{img.shape[0]}")

    img_bgr = (
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if input_path.suffix.lower() in [".tif", ".tiff"]
        else img
    )

    print(f"   Loading Real-ESRGAN model...")
    model = RealESRGAN(scale=scale, tile_size=256)

    print(f"   Enhancing (this may take a while on CPU)...")
    output_bgr = model.enhance(img_bgr)

    print(f"   Output size: {output_bgr.shape[1]}x{output_bgr.shape[0]}")

    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if transform is not None:
        new_transform = Affine(
            transform.a / scale,
            transform.b,
            transform.c,
            transform.d,
            transform.e / scale,
            transform.f,
        )

        output_tif = output_path.with_suffix(".tif")
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=output_rgb.shape[0],
            width=output_rgb.shape[1],
            count=3,
            dtype="uint8",
            crs=crs,
            transform=new_transform,
            compress="lzw",
        ) as dst:
            for i in range(3):
                dst.write(output_rgb[:, :, i], i + 1)

        print(f"   ✅ Saved: {output_tif}")
        final_path = output_tif
    else:
        output_png = output_path.with_suffix(".png")
        cv2.imwrite(str(output_png), output_bgr)
        print(f"   ✅ Saved: {output_png}")
        final_path = output_png

    metadata = {
        "model": f"RealESRGAN_x{scale}",
        "scale": scale,
        "input_size": [img.shape[1], img.shape[0]],
        "output_size": [output_rgb.shape[1], output_rgb.shape[0]],
        "device": str(model.device),
        "original_resolution_m": 10.0,
        "effective_resolution_m": 10.0 / scale,
    }

    return final_path, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CNN Super-Resolution")
    parser.add_argument("input", help="Input image")
    parser.add_argument(
        "-o", "--output", default="./cnn_sr_output", help="Output directory"
    )
    parser.add_argument("-s", "--scale", type=int, choices=[2, 4], default=4)

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"cnn_sr_{timestamp}"

    result_path, metadata = apply_cnn_sr(
        Path(args.input),
        output_path,
        args.scale,
    )

    print(f"\nOutput: {result_path}")
    print(f"Metadata: {metadata}")

