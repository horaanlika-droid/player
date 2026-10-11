#!/usr/bin/env python3
"""Генерирует бесшовную зернистую текстуру фона под тон чёрного фона фото.

Статистика тёмных краёв IMG_1468.png: средняя яркость ~9-12, зерно std ~2.
Текстура кладётся поверх чёрного фона в CSS (body::before) с повторением,
поэтому вокруг плеера на планшетах виден тот же тон и то же зерно, что и в
самом фото — граница «фото / страница» исчезает.

Полутон подобрана так, чтобы итог совпал с фото:
    mean = 128 * 23/255 ≈ 11.5,  std = 22 * 23/255 ≈ 2.0

На выходе: assets/layers/noise.png (192x192 RGBA, ~30 КБ).
"""

import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'assets', 'layers', 'noise.png')
SIZE = 192
GRAY_MEAN, GRAY_SD = 128.0, 22.0
ALPHA = 23  # из 255 — итоговый тон ≈ #0b0b0b с зерном ±2


def build(out_path: str) -> None:
    rng = np.random.default_rng(22)  # фиксированный seed — файл воспроизводим
    g = rng.normal(GRAY_MEAN, GRAY_SD, (SIZE, SIZE)).clip(0, 255).astype(np.uint8)
    a = np.full((SIZE, SIZE), ALPHA, np.uint8)
    img = Image.fromarray(np.dstack([g, g, g, a]), 'RGBA')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, optimize=True)
    print(f'{out_path} ({SIZE}x{SIZE}, mean {GRAY_MEAN * ALPHA / 255:.1f}, '
          f'std {GRAY_SD * ALPHA / 255:.1f})')


if __name__ == '__main__':
    build(OUT)
