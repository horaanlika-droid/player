#!/usr/bin/env python3
"""Убирает старую бумажную наклейку «I ♥ 22» и звёздочку с веб-панели плеера.

Запускать после make_ambient.py: исходные IMG_1468.png и IMG_1468_cord.png
остаются архивными фотографиями, а на странице используется plate.webp.
Координаты заданы для исходных кадров 1024x1280 и 1024x1536.

Наклейка с бокалом («1 YEAR! CATCH 22») со страницы убрана целиком, поэтому
прозрачный спрайт из IMG_1448.jpeg больше не вырезается и не генерируется.
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
ORIG_PLAYER = ROOT / 'IMG_1468.png'
PLATE = ROOT / 'assets/layers/plate.webp'


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
    clean_plate()
