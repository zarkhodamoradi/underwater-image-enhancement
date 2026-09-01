"""
Batch classical CV enhancement pipeline for underwater images.

Refactors the notebook's per-image steps (gray-world white balance ->
CLAHE (in LAB space) -> gamma correction -> denoising -> sharpening)
into reusable functions, then applies them to every image in a folder.

Usage:
    python batch_preprocess.py --input raw-890 --output enhanced-890
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def gray_world_white_balance(img: np.ndarray, max_gain: float = 1.8) -> np.ndarray:
    """Gray-world white balance with gain capping. Without the cap, a channel
    with a very low mean (typically red, underwater) gets a huge scale factor
    and blows out into a color cast - capping keeps corrections plausible."""
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    scale_b = np.clip(mean_gray / (mean_b + 1e-5), 1 / max_gain, max_gain)
    scale_g = np.clip(mean_gray / (mean_g + 1e-5), 1 / max_gain, max_gain)
    scale_r = np.clip(mean_gray / (mean_r + 1e-5), 1 / max_gain, max_gain)
    b, g, r = b * scale_b, g * scale_g, r * scale_r
    balanced = cv2.merge([np.clip(c, 0, 255) for c in (b, g, r)])
    return balanced.astype(np.uint8)

def clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB space (avoids the color-shift
    you get applying CLAHE independently to B/G/R)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def gamma_correction(img: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    lut = np.array(
        [np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, lut)


def denoise(img: np.ndarray, h: int = 6, template=3, search=15) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=h, hColor=h, templateWindowSize=template, searchWindowSize=search
    )


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - (1 / 9) * np.ones((3, 3))
    return cv2.filter2D(img, -1, kernel)


def enhance(img: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Full pipeline: white balance -> CLAHE -> gamma -> denoise -> sharpen."""
    out = gray_world_white_balance(img)
    out = clahe_lab(out)
    out = gamma_correction(out, gamma=gamma)
    out = denoise(out)
    out = sharpen(out)
    return out


def run_batch(input_dir: Path, output_dir: Path, ext: str = "*.png") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(input_dir.glob(ext))
    if not image_paths:
        print(f"No images matching {ext} found in {input_dir}")
        return

    failed = []
    for path in tqdm(image_paths, desc="Enhancing"):
        img = cv2.imread(str(path))
        if img is None:
            failed.append(path.name)
            continue
        result = enhance(img)
        cv2.imwrite(str(output_dir / path.name), result)

    print(f"Done. {len(image_paths) - len(failed)}/{len(image_paths)} images processed.")
    if failed:
        print(f"Failed to read: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="raw-890")
    parser.add_argument("--output", type=str, default="enhanced-890")
    parser.add_argument("--ext", type=str, default="*.png")
    args = parser.parse_args()

    run_batch(Path(args.input), Path(args.output), args.ext)
