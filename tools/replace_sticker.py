#!/usr/bin/env python3
"""Вырезает наклейку с бокалом из IMG_1448.jpeg и убирает «I ♥ 22» с веб-панели.

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
PLATE = ROOT / 'assets/layers/plate.webp'
STICKER = ROOT / 'assets/layers/goblet-sticker.png'


def cut_sticker():
    photo = np.asarray(Image.open(PHOTO).convert('RGB'))
    if photo.shape[:2] != (1280, 1024):
        raise ValueError('IMG_1448.jpeg должен быть 1024x1280')
    # На верхнем листе бокал не заслонён другими наклейками. Цвет бумаги —
    # холодный голубой, сам бокал — оранжевый с почти чёрной обводкой.
    crop = photo[352:504, 407:570]
    r, g, b = (crop[:, :, i].astype(np.int16) for i in range(3))
    foreground = (((r > b + 22) & (r > 125) & (g > 80)) |
                  (np.max(crop, axis=2) < 105)).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                  np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    if count < 2:
        raise ValueError('Не удалось выделить наклейку с бокалом')
    goblet = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    alpha = cv2.GaussianBlur((labels == goblet).astype(np.uint8) * 255,
                             (3, 3), 0.5)
    ys, xs = np.where(alpha > 0)
    sticker = np.dstack((crop, alpha))[ys.min():ys.max()+1, xs.min():xs.max()+1]
    Image.fromarray(sticker, 'RGBA').save(STICKER, optimize=True)
    print(f'Наклейка: {STICKER} ({sticker.shape[1]}x{sticker.shape[0]})')


def clean_plate():
    plate = np.asarray(Image.open(PLATE).convert('RGBA'))
    if plate.shape[:2] != (1536, 1024):
        raise ValueError('plate.webp должен быть 1024x1536')
    plate = plate.copy()
    # Контур старой рваной бумажной наклейки. Соседние участки корпуса и
    # нижнюю окантовку окна кассеты не меняем.
    outline = np.array([
        (741, 521), (829, 519), (850, 530), (852, 549), (851, 609),
        (846, 630), (836, 650), (728, 651), (721, 638), (720, 608),
        (723, 584), (719, 557), (725, 536)
    ], dtype=np.int32)
    mask = np.zeros(plate.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outline], 255)
    # Сверху — мягкое продолжение тёмного фона; снизу копируем фактуру
    # кассеты с той же высоты слева, чтобы линии корпуса остались ровными.
    upper = cv2.inpaint(plate[:, :, :3], mask, 7, cv2.INPAINT_TELEA)
    result = plate.copy()
    y = np.arange(515, 657)[:, None]
    x = np.arange(710, 860)[None, :]
    donor_y = np.where(y < 552, y + 38, y)
    donor_x = np.where(y < 552, x - 270, x - 166)
    result[515:657, 710:860, :3] = plate[donor_y, donor_x, :3]
    fade = np.clip((y - 549) / 31, 0, 1)[..., None]
    result[515:580, 710:860, :3] = (
        (1 - fade[:65]) * upper[515:580, 710:860] +
        fade[:65] * result[515:580, 710:860, :3]
    ).astype(np.uint8)
    feather = cv2.GaussianBlur(mask, (0, 0), 2.2)[..., None] / 255.0
    plate[:, :, :3] = (
        feather * result[:, :, :3] + (1 - feather) * plate[:, :, :3]
    ).astype(np.uint8)
    # Звёздочка справа от CATCH22 тоже напечатана прямо на фотографии.
    # Маскируем лишь её светлые пиксели, не трогая сам логотип и фактуру.
    star_mask = np.zeros(plate.shape[:2], dtype=np.uint8)
    star_gray = cv2.cvtColor(plate[177:208, 819:853, :3], cv2.COLOR_RGB2GRAY)
    star_mask[177:208, 819:853] = (star_gray > 60).astype(np.uint8) * 255
    star_mask = cv2.dilate(star_mask, np.ones((5, 5), np.uint8))
    plate[:, :, :3] = cv2.inpaint(plate[:, :, :3], star_mask, 4,
                                  cv2.INPAINT_TELEA)
    Image.fromarray(plate, 'RGBA').save(PLATE, 'WEBP', quality=86,
                                         alpha_quality=90, method=6)
    print(f'Убрана старая наклейка: {PLATE}')


if __name__ == '__main__':
    cut_sticker()
    clean_plate()
