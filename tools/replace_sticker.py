#!/usr/bin/env python3
"""Вырезает и реставрирует наклейку с бокалом из IMG_1448.jpeg и убирает «I ♥ 22» с веб-панели.

Запускать после make_ambient.py: исходные IMG_1468.png и IMG_1468_cord.png
остаются архивными фотографиями, а на странице используется plate.webp.
Координаты заданы для исходных кадров 1024x1280 и 1024x1536.
"""

from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        f'Не хватает модуля {exc.name}. Выполните: python3 -m venv .venv && '
        '.venv/bin/pip install -r tools/requirements.txt'
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
PHOTO = ROOT / 'IMG_1448.jpeg'
ORIG_PLAYER = ROOT / 'IMG_1468.png'
PLATE = ROOT / 'assets/layers/plate.webp'
STICKER = ROOT / 'assets/layers/goblet-sticker.png'


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cut_sticker() -> None:
    photo = np.asarray(Image.open(PHOTO).convert('RGB'))
    if photo.shape[:2] != (1280, 1024):
        raise ValueError('IMG_1448.jpeg должен быть 1024x1280')
    orig_player = np.asarray(Image.open(ORIG_PLAYER).convert('RGB'))

    # Используем нижний лист из IMG_1448.jpeg: он ближе к объективу, крупнее
    # по разрешению и без пересвета на оранжевой заливке и шрифте.
    gob2 = photo[695:935, 685:925]
    r, g, b = (gob2[:, :, i].astype(np.int16) for i in range(3))
    orange2 = ((r - b > 32) & (r > 95) & (g > 60)).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(orange2)
    if count < 2:
        raise ValueError('Не удалось выделить наклейку с бокалом')
    main_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    orange_body = (labels == main_lbl).astype(np.uint8) * 255

    # Заполняем буквы внутри чаши бокала, чтобы получить чистый силуэт внутренней заливки
    h, w = orange_body.shape
    flood = orange_body.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)
    gob2_int = cv2.bitwise_or(orange_body, cv2.bitwise_not(flood)).astype(np.float32) / 255.0

    ys, xs = np.where(gob2_int > 0.5)
    c_in = np.array([xs.mean(), ys.mean()])

    # Ортографическая ректификация перспективы листа наклеек (масштаб 1.9x на холсте 420x420)
    h_canv = 420
    angle_deg = 45.54
    sy = 1.0846
    sx = 1.0 / sy
    shx = -0.0009
    scale = 1.9
    c_out = np.array([h_canv / 2.0, h_canv / 2.0])
    rad = -np.deg2rad(angle_deg)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    stretch = np.array([[sx, shx], [0.0, sy]]) * scale
    affine = rot @ stretch
    offset = c_out - affine @ c_in
    mat = np.hstack([affine, offset[:, None]]).astype(np.float32)

    gob2_rect = cv2.warpAffine(
        gob2, mat, (h_canv, h_canv),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )
    int_rect = cv2.warpAffine(gob2_int, mat, (h_canv, h_canv), flags=cv2.INTER_CUBIC)

    # Симметризованное знаковое поле расстояний (SDF) от контура чаши и ножки бокала
    int_sym = 0.5 * (int_rect + int_rect[:, ::-1])
    int_sym_blur = cv2.GaussianBlur(int_sym, (0, 0), 2.2)
    inside_u8 = (int_sym_blur > 0.5).astype(np.uint8)
    outside_u8 = 1 - inside_u8
    dist = (
        cv2.distanceTransform(outside_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        - cv2.distanceTransform(inside_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    )
    dist = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), 0.9)

    # Выделяем печатный рисунок букв «1 / YEAR! / CATCH / 22» с сохранением тонких
    # внутренних просветов шрифта (white top-hat)
    c_sig = (
        0.55 * gob2_rect[:, :, 0].astype(np.float32)
        + 0.35 * gob2_rect[:, :, 1].astype(np.float32)
        + 0.10 * gob2_rect[:, :, 2].astype(np.float32)
    )
    c_blur = cv2.GaussianBlur(c_sig, (0, 0), 1.0)
    c_sharp = c_sig + 0.75 * (c_sig - c_blur)
    kernel_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = c_sig - cv2.morphologyEx(c_sig, cv2.MORPH_OPEN, kernel_th)
    c_enhanced = c_sharp + 0.85 * np.maximum(0.0, tophat - 4.0)

    ink_low, orange_high = np.percentile(c_enhanced[dist < -4], [8, 76])
    t_orange = np.clip((c_enhanced - ink_low) / (orange_high - ink_low), 0.0, 1.0)
    t_orange = np.clip(0.5 + 0.5 * np.tanh((t_orange - 0.44) * 5.0), 0.0, 1.0)

    # Поворачиваем бокал на естественный угол наклейки на кассете (~29° по часовой стрелке)
    tilt_deg = -29.0
    m_tilt = cv2.getRotationMatrix2D((h_canv / 2.0, h_canv / 2.0), tilt_deg, 1.0)
    dist_t = cv2.warpAffine(dist, m_tilt, (h_canv, h_canv), flags=cv2.INTER_CUBIC, borderValue=100.0)
    t_orange_t = cv2.warpAffine(t_orange, m_tilt, (h_canv, h_canv), flags=cv2.INTER_CUBIC, borderValue=0.0)

    # Концентрические контуры вырубки и двойной обводки (по профилю наклеек на IMG_1468.png)
    rng = np.random.default_rng(22)
    noise_low = cv2.GaussianBlur(rng.standard_normal((h_canv, h_canv)).astype(np.float32), (0, 0), 7.5)
    noise_mid = cv2.GaussianBlur(rng.standard_normal((h_canv, h_canv)).astype(np.float32), (0, 0), 2.2)
    edge_jitter = (
        (noise_low / (noise_low.std() + 1e-5)) * 0.48
        + (noise_mid / (noise_mid.std() + 1e-5)) * 0.28
    )
    dist_cut = dist_t + edge_jitter

    w_inside = 1.0 - _smoothstep(-0.65, 0.65, dist_t)
    w_gap = _smoothstep(6.1, 7.1, dist_t) * (1.0 - _smoothstep(8.0, 9.0, dist_t))
    w_rim = _smoothstep(11.6, 12.8, dist_t) * (1.0 - _smoothstep(17.5, 18.9, dist_cut))
    sticker_alpha = 1.0 - _smoothstep(17.6, 19.0, dist_cut)

    # Изотропная фактура кракелюра и бумажного волокна из реальной наклейки IMG_1468.png
    paper_patch = orig_player[338:422, 220:640, :3].astype(np.float32).mean(axis=2)
    paper_hp = paper_patch - cv2.GaussianBlur(paper_patch, (0, 0), 4.0)
    strips = [paper_hp if i % 2 == 0 else paper_hp[::-1, ::-1] for i in range(5)]
    tex_real = np.vstack(strips)[:h_canv, :h_canv]
    tex_real = cv2.GaussianBlur(tex_real, (0, 0), 0.8)
    tex_real = (tex_real - tex_real.mean()) / (tex_real.std() + 1e-5)

    grain_fine = cv2.GaussianBlur(rng.standard_normal((h_canv, h_canv)).astype(np.float32), (0, 0), 0.7)
    grain_fine = grain_fine / (grain_fine.std() + 1e-5)
    blotch = cv2.GaussianBlur(rng.standard_normal((h_canv, h_canv)).astype(np.float32), (0, 0), 7.0)
    blotch = blotch / (blotch.std() + 1e-5)

    yy, xx = np.mgrid[0:h_canv, 0:h_canv].astype(np.float32)
    u = (xx - h_canv / 2.0) / 140.0
    v = (yy - h_canv / 2.0) / 140.0
    light = np.clip(-0.48 * u - 0.52 * v, -1.0, 1.0)
    vig = np.clip(1.0 - 0.09 * (u ** 2 + v ** 2), 0.85, 1.05)

    tex_mix = 0.55 * tex_real + 0.45 * grain_fine

    # Цветокоррекция под студийный свет плеера (в тон «i'm catch!» и банана на IMG_1468.png)
    orange_rgb = np.zeros((h_canv, h_canv, 3), dtype=np.float32)
    orange_rgb[:, :, 0] = (160.0 + 16.0 * light + 5.5 * blotch + 4.8 * tex_mix) * vig
    orange_rgb[:, :, 1] = (90.5 + 10.0 * light + 3.3 * blotch + 3.0 * tex_mix) * vig
    orange_rgb[:, :, 2] = (29.0 + 4.5 * light + 1.4 * blotch + 1.4 * tex_mix) * vig

    ink_rgb = np.zeros((h_canv, h_canv, 3), dtype=np.float32)
    ink_mod = np.clip(13.0 + 2.4 * light + 1.2 * blotch + 1.8 * np.maximum(0.0, tex_mix), 6.5, 30.0)
    ink_rgb[:, :, 0] = ink_mod * 1.03
    ink_rgb[:, :, 1] = ink_mod * 1.00
    ink_rgb[:, :, 2] = ink_mod * 0.88

    gap_rgb = np.zeros((h_canv, h_canv, 3), dtype=np.float32)
    gap_mod = 68.0 + 9.0 * light + 2.5 * blotch + 2.5 * tex_mix
    gap_rgb[:, :, 0] = gap_mod * 1.04
    gap_rgb[:, :, 1] = gap_mod * 0.97
    gap_rgb[:, :, 2] = gap_mod * 0.87

    rim_rgb = np.zeros((h_canv, h_canv, 3), dtype=np.float32)
    rim_mod = 138.0 + 17.0 * light + 5.5 * blotch + 4.8 * tex_mix
    rim_rgb[:, :, 0] = rim_mod * 1.05
    rim_rgb[:, :, 1] = rim_mod * 0.96
    rim_rgb[:, :, 2] = rim_mod * 0.86

    rgb = ink_rgb.copy()
    w_or = (w_inside * t_orange_t)[..., None]
    rgb = (1.0 - w_or) * rgb + w_or * orange_rgb
    rgb = (1.0 - w_gap[..., None]) * rgb + w_gap[..., None] * gap_rgb
    rgb = (1.0 - w_rim[..., None]) * rgb + w_rim[..., None] * rim_rgb

    # Лёгкие потёртости бумажного канта и фаска толщины вырубки
    scuff_noise = cv2.GaussianBlur(rng.standard_normal((h_canv, h_canv)).astype(np.float32), (0, 0), 2.4)
    scuff_mask = np.clip((scuff_noise / (scuff_noise.std() + 1e-5)) - 1.08, 0.0, 1.6) * w_rim
    rgb *= (1.0 - 0.35 * scuff_mask[..., None])

    gy, gx = np.gradient(dist_cut)
    gnorm = np.hypot(gx, gy) + 1e-5
    nx, ny = gx / gnorm, gy / gnorm
    rim_edge_band = np.exp(-((dist_cut - 17.3) / 1.0) ** 2)
    ndotl_edge = nx * (-0.65) + ny * (-0.76)
    bevel_hi = np.maximum(0.0, ndotl_edge) * rim_edge_band * 22.0
    bevel_sh = np.maximum(0.0, -ndotl_edge) * rim_edge_band * 28.0
    rgb += bevel_hi[..., None] * np.array([1.05, 0.97, 0.88])[None, None, :]
    rgb -= bevel_sh[..., None]

    rgba = np.dstack([
        np.clip(np.round(rgb), 0, 255).astype(np.uint8),
        np.clip(np.round(sticker_alpha * 255.0), 0, 255).astype(np.uint8),
    ])
    ys_a, xs_a = np.where(rgba[:, :, 3] > 2)
    pad = 3
    y0_a, y1_a = max(0, int(ys_a.min()) - pad), min(h_canv, int(ys_a.max()) + 1 + pad)
    x0_a, x1_a = max(0, int(xs_a.min()) - pad), min(h_canv, int(xs_a.max()) + 1 + pad)
    sticker_crop = rgba[y0_a:y1_a, x0_a:x1_a]

    w_save = 240
    h_save = int(round(w_save * sticker_crop.shape[0] / sticker_crop.shape[1]))
    sticker_final = cv2.resize(sticker_crop, (w_save, h_save), interpolation=cv2.INTER_AREA)
    Image.fromarray(sticker_final, 'RGBA').save(STICKER, optimize=True)
    print(f'Наклейка: {STICKER} ({sticker_final.shape[1]}x{sticker_final.shape[0]})')


def clean_plate() -> None:
    plate = np.asarray(Image.open(PLATE).convert('RGBA'))
    if plate.shape[:2] != (1536, 1024):
        raise ValueError('plate.webp должен быть 1024x1536')
    orig = np.asarray(Image.open(ORIG_PLAYER).convert('RGBA'))
    plate = plate.copy()

    # Контур старой рваной бумажной наклейки «I ♥ 22». Восстанавливаем из чистого
    # участка корпуса кассеты сверху и слева без размытия фактуры пластика.
    outline = np.array([
        (741, 521), (829, 519), (850, 530), (852, 549), (851, 609),
        (846, 630), (836, 650), (728, 651), (721, 638), (720, 608),
        (723, 584), (719, 557), (725, 536)
    ], dtype=np.int32)
    mask = np.zeros(plate.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outline], 255)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    result = orig.copy()
    result[515:581, 710:860, :3] = orig[444:510, 710:860, :3]
    result[581:657, 710:806, :3] = orig[581:657, 544:640, :3]
    result[581:657, 800:860, :3] = orig[441:517, 800:860, :3]

    w_x = np.linspace(0.0, 1.0, 12)[None, :, None]
    result[581:657, 796:808, :3] = (
        (1.0 - w_x) * orig[581:657, 630:642, :3]
        + w_x * orig[441:517, 796:808, :3]
    ).astype(np.uint8)

    w_y = np.linspace(0.0, 1.0, 8)[:, None, None]
    result[577:585, 710:808, :3] = (
        (1.0 - w_y) * orig[506:514, 710:808, :3]
        + w_y * result[577:585, 710:808, :3]
    ).astype(np.uint8)

    feather = cv2.GaussianBlur(mask, (0, 0), 2.5)[..., None] / 255.0
    plate[:, :, :3] = (
        feather * result[:, :, :3] + (1.0 - feather) * plate[:, :, :3]
    ).astype(np.uint8)

    # Звёздочка справа от CATCH22 тоже напечатана прямо на фотографии.
    star_mask = np.zeros(plate.shape[:2], dtype=np.uint8)
    star_gray = cv2.cvtColor(orig[177:208, 819:853, :3], cv2.COLOR_RGB2GRAY)
    star_mask[177:208, 819:853] = (star_gray > 60).astype(np.uint8) * 255
    star_mask = cv2.dilate(star_mask, np.ones((5, 5), np.uint8))
    plate[:, :, :3] = cv2.inpaint(
        plate[:, :, :3], star_mask, 4, cv2.INPAINT_TELEA
    )
    Image.fromarray(plate, 'RGBA').save(
        PLATE, 'WEBP', quality=86, alpha_quality=90, method=6
    )
    print(f'Убрана старая наклейка: {PLATE}')


if __name__ == '__main__':
    cut_sticker()
    clean_plate()
