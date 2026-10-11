#!/usr/bin/env python3
"""Продлевает матовый тёмный фон IMG_1468.png на весь экран без видимой рамки.

Скрипт решает две задачи:
1. Строит бесшовный тайл фона `assets/layers/noise.png` (256x256 RGB) методом
   спектрального синтеза (2D FFT) по реальному участку матового фона из
   верхней части `IMG_1468.png` (y=0..96, x=240..880). Сохраняются точный
   тёплый оливково-угольный тон (RGB ≈ [10.9, 11.1, 9.8], #0b0b0a), разброс
   яркости (std ≈ 2.0) и пространственная корреляция зерна (lag 1 ≈ 0.55,
   lag 2 ≈ 0.11). За счёт периодичности БПФ тайл стыкуется без швов.
2. Обновляет `assets/layers/plate.webp` (1024x1536 RGBA): корпус плеера,
   боковой блок регуляторов и штекер наушников остаются полностью непрозрачными
   (alpha = 255), а внешние поля кадра вокруг силуэта плеера плавно сводятся
   к alpha = 0 (cubic smoothstep на полосе ~36 px) с выравниванием тона
   к базовому фону. В результате у картинки исчезает прямоугольная граница
   1024x1536, а CSS drop-shadow на `.device` ложится по реальному силуэту
   плеера, а не вокруг прямоугольника фотографии.
"""

import os

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:  # подсказка вместо голого traceback
    raise SystemExit(
        'Инструментам нужны зависимости (не хватает модуля: %s). Выполните:\n'
        '  python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt\n'
        'и запускайте скрипт через .venv/bin/python' % exc.name
    ) from exc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'IMG_1468.png')
OUT_NOISE = os.path.join(ROOT, 'assets', 'layers', 'noise.png')
OUT_PLATE = os.path.join(ROOT, 'assets', 'layers', 'plate.webp')

SIZE = 256
MAX_LAG = 5
BASE_RGB = np.array([10.9, 11.1, 9.8], dtype=np.float64)
GRAIN_STD = 2.0


def sd_round_rect(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    r: float,
) -> np.ndarray:
    """Знаковое расстояние до скруглённого прямоугольника [x0..x1] x [y0..y1]."""
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    hx = (x1 - x0) * 0.5 - r
    hy = (y1 - y0) * 0.5 - r
    qx = np.abs(x - cx) - hx
    qy = np.abs(y - cy) - hy
    dx = np.maximum(qx, 0.0)
    dy = np.maximum(qy, 0.0)
    return np.hypot(dx, dy) + np.minimum(np.maximum(qx, qy), 0.0) - r


def build_noise(src: np.ndarray, out_path: str) -> None:
    """Генерирует бесшовный 256x256 RGB-тайл с тем же спектром зерна, что у фото."""
    patch = src[0:96, 240:880, 1]
    pad = np.pad(patch, 6, mode='reflect')
    low = np.mean(
        [[pad[i:i + 96, j:j + 640] for i in range(13)] for j in range(13)],
        axis=(0, 1),
    )
    hp = patch - low

    ac = np.zeros((SIZE, SIZE), dtype=np.float64)
    h_p, w_p = hp.shape
    for dy in range(-MAX_LAG, MAX_LAG + 1):
        for dx in range(-MAX_LAG, MAX_LAG + 1):
            y0, y1 = max(0, dy), min(h_p, h_p + dy)
            x0, x1 = max(0, dx), min(w_p, w_p + dx)
            c = (hp[y0:y1, x0:x1] * hp[y0 - dy:y1 - dy, x0 - dx:x1 - dx]).mean()
            wy = 0.5 * (1.0 + np.cos(np.pi * dy / (MAX_LAG + 1)))
            wx = 0.5 * (1.0 + np.cos(np.pi * dx / (MAX_LAG + 1)))
            ac[dy % SIZE, dx % SIZE] = c * wy * wx

    amp = np.sqrt(np.maximum(np.fft.fft2(ac).real, 0.0))
    rng = np.random.default_rng(22)
    grain = np.fft.ifft2(np.fft.fft2(rng.standard_normal((SIZE, SIZE))) * amp).real
    grain = (grain - grain.mean()) / grain.std() * GRAIN_STD

    tile_rgb = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    for ch in range(3):
        ch_n = rng.standard_normal((SIZE, SIZE)) * 0.25
        tile_rgb[:, :, ch] = BASE_RGB[ch] + grain + ch_n

    tile_u8 = np.clip(np.round(tile_rgb), 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(tile_u8, 'RGB').save(out_path, optimize=True)
    print(
        f'noise: {out_path} ({SIZE}x{SIZE} RGB, '
        f'mean={tile_u8.mean(axis=(0, 1)).round(2).tolist()}, '
        f'std={tile_u8.std(axis=(0, 1)).round(2).tolist()})'
    )


def build_plate(src: np.ndarray, out_path: str) -> None:
    """Сохраняет plate.webp с мягким сведением внешних полей фона в прозрачность."""
    h, w, _ = src.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Силуэт плеера: основной корпус, боковой блок регуляторов и штекер наушников
    d_body = sd_round_rect(xx, yy, 65, 102, 912, 1496, 42)
    d_side = sd_round_rect(xx, yy, 890, 234, 1001, 1142, 16)
    d_plug_barrel = sd_round_rect(xx, yy, 139, 26, 212, 120, 20)
    d_plug_tip = sd_round_rect(xx, yy, 158, 2, 197, 45, 10)
    d_player = np.minimum(
        np.minimum(d_body, d_side),
        np.minimum(d_plug_barrel, d_plug_tip),
    )

    d_edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    t_halo = np.clip(1.0 - np.maximum(d_player, 0.0) / 36.0, 0.0, 1.0)
    t_edge = np.clip((d_edge - 2.0) / 18.0, 0.0, 1.0)
    t = t_halo * t_edge
    alpha_f = t * t * (3.0 - 2.0 * t)
    alpha = np.round(alpha_f * 255.0).astype(np.uint8)

    src_mod = src.copy()
    # Мягкое угасание самого верхнего среза провода штекера (y=0..14), чтобы не было
    # горизонтального обрыва у верхнего края кадра
    plug_fade = np.clip((yy - 2.0) / 12.0, 0.0, 1.0)[:, :, None]
    src_mod[:16, 150:205, :] = (
        src_mod[:16, 150:205, :] * plug_fade[:16, 150:205, :]
        + BASE_RGB * (1.0 - plug_fade[:16, 150:205, :])
    )
    # В зоне полного растворения фиксируем базовый тон, чтобы WebP не давал ореолов
    src_mod[alpha == 0] = BASE_RGB

    plate_rgba = np.dstack([src_mod.clip(0, 255).astype(np.uint8), alpha])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(plate_rgba, 'RGBA').save(
        out_path, 'WEBP', quality=86, alpha_quality=90, method=6
    )
    print(f'plate: {out_path} ({w}x{h} RGBA WebP, border max alpha={alpha[0, :].max()})')


def main() -> None:
    src = np.asarray(Image.open(SRC).convert('RGB'), dtype=np.float64)
    build_noise(src, OUT_NOISE)
    build_plate(src, OUT_PLATE)


if __name__ == '__main__':
    main()
