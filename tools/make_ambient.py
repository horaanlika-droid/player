#!/usr/bin/env python3
"""Продлевает матовый тёмный фон IMG_1468.png и прямой чёрный провод штекера.

Скрипт решает четыре задачи:
1. Строит бесшовный тайл фона `assets/layers/noise.png` (256x256 RGB) методом
   спектрального синтеза (2D FFT) по реальному участку матового фона из
   верхней части `IMG_1468.png` (y=0..96, x=240..880). Сохраняются точный
   тёплый оливково-угольный тон (RGB ≈ [10.9, 11.1, 9.8], #0b0b0a), разброс
   яркости (std ≈ 2.0) и пространственная корреляция зерна (lag 1 ≈ 0.55,
   lag 2 ≈ 0.11). За счёт периодичности БПФ тайл стыкуется без швов.
2. Строит бесшовный по вертикали RGBA-спрайт прямого чёрного провода наушников
   `assets/layers/cord.png` (64x256 RGBA, x=146..210 в системе координат
   1024x1536, ось провода x=177.0, диаметр ~23.6 px — точно по верхнему плечу
   хвостовика штекера PHONES). Цилиндрическая светотень с передне-левым бликом,
   глубокой собственной тенью справа и периодическим 2D-FFT микрозерном матовой
   резины позволяет `.cord` повторяться вверх на любую высоту экрана и уходить
   за верхний предел фона и окна.
3. Обновляет `assets/layers/plate.webp` (1024x1536 RGBA): убирает обрезанные
   верхней кромкой кадра «рожки» четвёртого ребра хвостовика на y=0..4,
   формирует аккуратную канавку выхода кабеля (y=4..5) над верхним ребром
   штекера и выводит из неё основание прямого чёрного провода до y=0 с тем же
   профилем RGB и альфа-маски, что у `cord.png`. Внешние поля вокруг силуэта
   плеера плавно сводятся в alpha = 0 (cubic smoothstep на полосе ~36 px).
4. Обновляет полный растровый кадр `IMG_1468_cord.png` (1024x1956 RGB, +420 px
   бесшовного матового фона сверху) с тем же прямым чёрным проводом, выходящим
   ровно из штекера PHONES и уходящим вверх за верхний предел фона (y=0).
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
OUT_CORD = os.path.join(ROOT, 'assets', 'layers', 'cord.png')
OUT_PLATE = os.path.join(ROOT, 'assets', 'layers', 'plate.webp')
OUT_FULL_CORD = os.path.join(ROOT, 'IMG_1468_cord.png')

SIZE = 256
MAX_LAG = 5
BASE_RGB = np.array([10.9, 11.1, 9.8], dtype=np.float64)
GRAIN_STD = 2.0

# Геометрия прямого чёрного провода в координатах кадра 1024x1536
CORD_X0 = 146      # левый край полосы 64 px (146 / 1024 = 14.2578125%)
CORD_W = 64        # ширина полосы (64 / 1024 = 6.25%)
CORD_H = 256       # период вертикального тайла
CORD_CX = 177.0    # ось хвостовика штекера PHONES
CORD_R = 11.8      # радиус провода, px (диаметр ~23.6 px, x ≈ 165.2..188.8)
EXT_TOP = 420      # высота продления фона над плеером в IMG_1468_cord.png


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


def build_noise(src: np.ndarray, out_path: str) -> np.ndarray:
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
    return tile_u8.astype(np.float64)


def cord_model(w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает 1D-профиль освещённости, амплитуды зерна, альфа-маски и 2D-текстуру резины."""
    rng = np.random.default_rng(22)
    raw = rng.standard_normal((CORD_H, CORD_W))
    fy = np.fft.fftfreq(CORD_H)[:, None]
    fx = np.fft.fftfreq(CORD_W)[None, :]
    freq = np.hypot(fx * 1.2, fy * 0.8)
    env = (
        np.exp(-((freq - 0.12) / 0.18) ** 2)
        + 1.4 * np.exp(-((freq - 0.04) / 0.06) ** 2)
    )
    rubber_tex = np.fft.ifft2(np.fft.fft2(raw) * env).real
    rubber_tex = (rubber_tex - rubber_tex.mean()) / rubber_tex.std()

    xx = np.arange(w, dtype=np.float64)[None, :]
    dx = xx - CORD_CX
    u = np.clip(dx / CORD_R, -1.0, 1.0)
    nz = np.sqrt(np.maximum(0.0, 1.0 - u ** 2))

    # Мягкая антиалиасинговая маска краёв цилиндрического провода
    cord_alpha = np.clip((CORD_R + 0.65 - np.abs(dx)) / 1.3, 0.0, 1.0)
    cord_alpha = cord_alpha * cord_alpha * (3.0 - 2.0 * cord_alpha)

    # Передне-левое освещение в тон штекера и рёбер амортизатора (пик x ≈ 173.8)
    lx = -0.28
    lz = np.sqrt(1.0 - lx ** 2)
    ndotl = np.maximum(0.0, u * lx + nz * lz)

    rim_left = np.exp(-((u + 0.85) / 0.11) ** 2) * 14.0
    dip_left = np.exp(-((u + 0.66) / 0.12) ** 2) * 5.0
    spec_broad = np.exp(-((u + 0.25) / 0.27) ** 2) * 28.0
    spec_peak = np.exp(-((u + 0.28) / 0.13) ** 2) * 22.0
    core_shadow = np.exp(-((u - 0.55) / 0.32) ** 2) * 8.5
    rim_right = np.exp(-((u - 0.84) / 0.10) ** 2) * 7.5

    base_lum = np.clip(
        4.5
        + 13.0 * (ndotl ** 1.5)
        + spec_broad
        + spec_peak
        + rim_left
        - dip_left
        - core_shadow
        + rim_right,
        1.2,
        72.0,
    )
    grain_amp = (0.8 + 1.8 * (base_lum / 60.0)) * nz
    return base_lum, grain_amp, cord_alpha, rubber_tex


def build_cord(
    base_lum: np.ndarray,
    grain_amp: np.ndarray,
    cord_alpha: np.ndarray,
    rubber_tex: np.ndarray,
    out_path: str,
) -> None:
    """Сохраняет вертикально бесшовный RGBA-спрайт провода (64x256)."""
    b_slice = base_lum[:, CORD_X0:CORD_X0 + CORD_W]
    g_slice = grain_amp[:, CORD_X0:CORD_X0 + CORD_W]
    a_slice = cord_alpha[:, CORD_X0:CORD_X0 + CORD_W]

    lum = np.clip(b_slice + rubber_tex * g_slice, 0.8, 85.0)
    rgb = np.stack([lum * 1.03, lum * 1.00, lum * 0.95], axis=-1)
    alpha = np.broadcast_to(a_slice, (CORD_H, CORD_W))

    rgba = np.dstack([
        np.clip(np.round(rgb), 0, 255).astype(np.uint8),
        np.round(alpha * 255.0).astype(np.uint8),
    ])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(rgba, 'RGBA').save(out_path, optimize=True)
    print(
        f'cord: {out_path} ({CORD_W}x{CORD_H} RGBA, '
        f'opaque cols={(np.where(a_slice[0] > 0.5)[0] + CORD_X0)[[0, -1]].tolist()})'
    )


def apply_cord_to_canvas(
    canvas: np.ndarray,
    bg_tile: np.ndarray,
    y_plug_top: int,
    base_lum: np.ndarray,
    grain_amp: np.ndarray,
    cord_alpha: np.ndarray,
    rubber_tex: np.ndarray,
) -> np.ndarray:
    """Дорисовывает прямой чёрный провод от верхнего ребра штекера (y_plug_top+5) до y=0."""
    h_zone = y_plug_top + 6
    w = canvas.shape[1]
    yy, xx = np.mgrid[0:h_zone, 0:w].astype(np.float64)
    dx = xx - CORD_CX

    # 1. Убираем срезанные верхней границей исходного фото «рожки» 4-го ребра на y=0..4
    for y_orig in range(5):
        y_c = y_plug_top + y_orig
        b = np.clip((4.6 - y_orig) / 1.0, 0.0, 1.0)
        bg_row = bg_tile[y_c % SIZE, np.arange(155, 202) % SIZE]
        canvas[y_c, 155:202] = (1.0 - b) * canvas[y_c, 155:202] + b * bg_row

    # 2. Мягкая канавка выхода кабеля (y_orig ≈ 4.2) и АО над верхним ребром штекера
    dy_collar = yy - (y_plug_top + 4.2)
    collar_groove = 1.0 - 0.72 * np.exp(-((dy_collar / 1.15) ** 2))
    collar_ao = 1.0 - 0.15 * np.exp(-np.maximum(0.0, (y_plug_top + 4.0) - yy) / 5.0)

    # Индекс текстуры согласован так, чтобы строка y=y_plug_top продолжала тайл cord.png (y=255)
    tex_y = (yy.astype(int) - y_plug_top) % CORD_H
    tex_x = (xx.astype(int) - CORD_X0) % CORD_W
    tex_mapped = rubber_tex[tex_y, tex_x]

    lum = np.clip(base_lum * collar_groove * collar_ao + tex_mapped * grain_amp, 0.8, 85.0)
    cord_rgb = np.stack([lum * 1.03, lum * 1.00, lum * 0.95], axis=-1)

    exit_blend = np.clip(((y_plug_top + 5.1) - yy) / 0.9, 0.0, 1.0)
    ca = cord_alpha * exit_blend

    # Лёгкая собственная тень на фоне рядом с проводом
    sh_left = np.exp(-(((dx + CORD_R + 1.8) / 3.0) ** 2)) * 3.0
    sh_right = np.exp(-(((dx - CORD_R - 4.0) / 6.5) ** 2)) * 5.5
    ambient_sh = (sh_left + sh_right) * exit_blend
    canvas[:h_zone] = np.maximum(0.0, canvas[:h_zone] - ambient_sh[..., None])

    canvas[:h_zone] = (1.0 - ca[..., None]) * canvas[:h_zone] + ca[..., None] * cord_rgb
    return ca


def build_plate(
    src: np.ndarray,
    bg_tile: np.ndarray,
    base_lum: np.ndarray,
    grain_amp: np.ndarray,
    cord_alpha: np.ndarray,
    rubber_tex: np.ndarray,
    out_path: str,
) -> np.ndarray:
    """Сохраняет plate.webp с мягкой альфа-маской корпуса и основанием провода до y=0."""
    h, w, _ = src.shape
    src_mod = src.copy()
    ca_top = apply_cord_to_canvas(
        src_mod, bg_tile, 0, base_lum, grain_amp, cord_alpha, rubber_tex
    )

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Силуэт плеера: основной корпус, боковой блок регуляторов, ствол и 3 ребра штекера
    d_body = sd_round_rect(xx, yy, 65, 102, 912, 1496, 42)
    d_side = sd_round_rect(xx, yy, 890, 234, 1001, 1142, 16)
    d_plug_barrel = sd_round_rect(xx, yy, 139, 26, 212, 120, 20)
    d_plug_ribs = sd_round_rect(xx, yy, 158, 4, 197, 45, 2)
    d_player = np.minimum(
        np.minimum(d_body, d_side),
        np.minimum(d_plug_barrel, d_plug_ribs),
    )

    d_edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    t_halo = np.clip(1.0 - np.maximum(d_player, 0.0) / 36.0, 0.0, 1.0)
    t_edge = np.clip((d_edge - 2.0) / 18.0, 0.0, 1.0)
    t = t_halo * t_edge
    alpha_f = t * t * (3.0 - 2.0 * t)

    # Рёбра и ствол штекера у верхнего края кадра остаются полностью непрозрачными,
    # а в зоне y=0..5 альфа провода выходит до y=0 строго по профилю cord.png
    plug_solid = np.clip(1.0 - np.maximum(np.minimum(d_plug_barrel, d_plug_ribs), 0.0) / 2.0, 0.0, 1.0)
    alpha_f = np.maximum(alpha_f, plug_solid)
    alpha_f[:6] = np.maximum(alpha_f[:6], ca_top)
    alpha = np.round(alpha_f * 255.0).astype(np.uint8)

    # В зоне полного растворения фиксируем базовый тон, чтобы WebP не давал ореолов
    src_mod[alpha == 0] = BASE_RGB

    plate_rgba = np.dstack([src_mod.clip(0, 255).astype(np.uint8), alpha])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(plate_rgba, 'RGBA').save(
        out_path, 'WEBP', quality=86, alpha_quality=90, method=6
    )
    print(f'plate: {out_path} ({w}x{h} RGBA WebP, border max alpha={alpha[0, :].max()})')
    return alpha_f


def build_full_cord_image(
    src: np.ndarray,
    bg_tile: np.ndarray,
    alpha_plate: np.ndarray,
    base_lum: np.ndarray,
    grain_amp: np.ndarray,
    cord_alpha: np.ndarray,
    rubber_tex: np.ndarray,
    out_path: str,
) -> None:
    """Строит IMG_1468_cord.png (1024x1956 RGB) с проводом, уходящим за верхний край фона."""
    h, w, _ = src.shape
    total_h = h + EXT_TOP
    yy, xx = np.mgrid[0:total_h, 0:w]
    canvas = bg_tile[yy % SIZE, xx % SIZE].copy()

    # Плавно вписываем силуэт плеера в бесшовный фон, чтобы не было горизонтального шва на y=EXT_TOP
    a_exp = alpha_plate[..., None]
    canvas[EXT_TOP:] = (1.0 - a_exp) * canvas[EXT_TOP:] + a_exp * src

    apply_cord_to_canvas(
        canvas, bg_tile, EXT_TOP, base_lum, grain_amp, cord_alpha, rubber_tex
    )

    out_u8 = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    Image.fromarray(out_u8, 'RGB').save(out_path, optimize=True)
    print(f'full_cord: {out_path} ({w}x{total_h} RGB)')


def main() -> None:
    src = np.asarray(Image.open(SRC).convert('RGB'), dtype=np.float64)
    bg_tile = build_noise(src, OUT_NOISE)
    base_lum, grain_amp, cord_alpha, rubber_tex = cord_model(src.shape[1])
    build_cord(base_lum, grain_amp, cord_alpha, rubber_tex, OUT_CORD)
    alpha_plate = build_plate(
        src, bg_tile, base_lum, grain_amp, cord_alpha, rubber_tex, OUT_PLATE
    )
    build_full_cord_image(
        src, bg_tile, alpha_plate, base_lum, grain_amp, cord_alpha, rubber_tex, OUT_FULL_CORD
    )


if __name__ == '__main__':
    main()
