"""Generates SYNTHETIC chest-X-ray-like images so the whole pipeline can be tested
before the real dataset is downloaded. NOT medical data.

    python backend/make_demo_data.py            # -> data/demo/{train,test}/{NORMAL,PNEUMONIA}
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def make_image(rng, pneumonia: bool, size: int = 256) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = 0.55 + 0.04 * rng.standard_normal((size, size))
    for cx in (0.32, 0.68):  # dark lungs
        img -= 0.35 * np.exp(-(((xx - cx) / 0.16) ** 2 + ((yy - 0.5) / 0.3) ** 2))
    img += 0.2 * np.exp(-((xx - 0.5) / 0.05) ** 2)  # spine
    if pneumonia:  # bright opacities inside lungs
        for _ in range(rng.integers(2, 5)):
            cx = rng.choice([0.32, 0.68]) + rng.normal(0, 0.04)
            cy, r = rng.uniform(0.35, 0.7), rng.uniform(0.04, 0.09)
            img += 0.4 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def main(n_train=150, n_test=40, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("test", n_test)):
        for cls in ("NORMAL", "PNEUMONIA"):
            d = ROOT / "data" / "demo" / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                make_image(rng, cls == "PNEUMONIA").save(d / f"{cls.lower()}_{i:03d}.png")
    print("Demo data written to", ROOT / "data" / "demo")


if __name__ == "__main__":
    main()
