#!/usr/bin/env python3
"""Аккуратно вырезает бобины (reels) кассеты из IMG_1468.png.

Центры хабов найдены Hough-голосованием по краям и уточнены
максимизацией 8-лучевой симметрии спиц. Режущий радиус 50 px
захватывает зубчатый хаб целиком (край зубцов ~47 px), но не
захватывает внешнее кольцо окна (51-55 px) с его фиксированным
бликом — кольцо остаётся на фото и не вращается.

На выходе: assets/layers/reel-a.png, assets/layers/reel-b.png —
квадратные RGBA-спрайты с мягкой круглой маской (фейдер 48-50 px).
"""

import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'IMG_1468.png')
OUT_DIR = os.path.join(ROOT, 'assets', 'layers')

# центры бобин в пикселях IMG_1468.png (1024x1536)
CENTERS = {
    'reel-a': (336.3, 491.7),
    'reel-b': (645.3, 491.7),
}
R_OUT = 50.0      # режущий радиус, px
R_FEATHER = 48.0  # до этого радиуса альфа = 1, дальше линейно до 0
SS = 4            # суперсэмплинг маски


def cut(src: str, out_dir: str) -> None:
    im = np.asarray(Image.open(src).convert('RGB'))
    os.makedirs(out_dir, exist_ok=True)
    size = int(2 * R_OUT)
    yy, xx = np.mgrid[0:size * SS, 0:size * SS]
    c = (size * SS - 1) / 2.0
    r = np.hypot(xx - c, yy - c) / SS
    alpha = np.clip((R_OUT - r) / (R_OUT - R_FEATHER), 0, 1)
    mask = Image.fromarray((alpha * 255).astype(np.uint8), 'L')
    mask = mask.resize((size, size), Image.LANCZOS)
    alpha = np.asarray(mask)

    for name, (cx, cy) in CENTERS.items():
        x0 = int(round(cx)) - size // 2
        y0 = int(round(cy)) - size // 2
        crop = im[y0:y0 + size, x0:x0 + size].copy()
        rgba = np.dstack([crop, alpha])
        path = os.path.join(out_dir, f'{name}.png')
        Image.fromarray(rgba, 'RGBA').save(path)
        print(f'{name}: {path} ({size}x{size}, центр {cx}, {cy})')


if __name__ == '__main__':
    cut(SRC, OUT_DIR)
