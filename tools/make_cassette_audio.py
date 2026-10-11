#!/usr/bin/env python3
"""Готовит фоновый трек assets/hbd.mp3 из 'Gorillaz – Feel Good Inc..mp3' с аналоговым кассетным характером.

Добавляет лёгкую кассетную шероховатость (Type I ferric):
- микро-детонацию протяжки (wow & flutter);
- подъём низких частот головки (~82 Гц) и спад ВЧ выше 11.2 кГц;
- предыскажения -> мягкое асимметричное насыщение магнитной ленты -> коррекцию;
- сигнально-зависимую модуляционную шероховатость оксидного слоя (asperity noise)
  и деликатный тёплый аналоговый шум дорожки.
"""

from pathlib import Path
import subprocess

try:
    import imageio_ffmpeg
    import numpy as np
    from scipy import signal
except ModuleNotFoundError as exc:
    raise SystemExit(
        f'Не хватает модуля {exc.name}. Выполните: python3 -m venv .venv && '
        '.venv/bin/pip install -r tools/requirements.txt'
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
SRC_MP3 = ROOT / 'Gorillaz – Feel Good Inc..mp3'
OUT_MP3 = ROOT / 'assets/hbd.mp3'
SR = 44100


def _peaking_eq(f0: float, q: float, gain_db: float, fs: int = SR) -> np.ndarray:
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * amp
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * amp
    a0 = 1.0 + alpha / amp
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / amp
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def build_cassette_track() -> None:
    if not SRC_MP3.exists():
        raise FileNotFoundError(f'Не найден исходный трек: {SRC_MP3}')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cmd_in = [
        ffmpeg, '-v', 'error', '-i', str(SRC_MP3),
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '2', '-ar', str(SR), '-',
    ]
    raw = subprocess.check_output(cmd_in)
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    n_samples = len(x)
    t = np.arange(n_samples, dtype=np.float64) / SR

    # 1. Запас по уровню перед виртуальной магнитной головкой
    x *= 0.78

    # 2. Органическая микро-детонация лентопротяжного механизма (wow & flutter)
    rng = np.random.default_rng(22)
    wow = (
        0.00018 * np.sin(2.0 * np.pi * 0.41 * t + 0.3)
        + 0.00011 * np.sin(2.0 * np.pi * 0.73 * t + 1.7)
        + 0.00004 * np.sin(2.0 * np.pi * 5.4 * t + 0.9)
    ) * SR
    idx = np.clip(np.arange(n_samples, dtype=np.float64) + wow, 0, n_samples - 2)
    idx_i = idx.astype(np.int64)
    frac = (idx - idx_i).astype(np.float32)[:, None]
    x = x[idx_i] * (1.0 - frac) + x[idx_i + 1] * frac

    # 3. НЧ-контур головки (срез ниже 38 Гц + тёплый подъём 82 Гц) и присутствие СЧ
    sos_hp = signal.butter(2, 38.0 / (SR / 2.0), btype='highpass', output='sos')
    x = signal.sosfilt(sos_hp, x, axis=0).astype(np.float32)
    x = signal.sosfilt(_peaking_eq(82.0, 1.1, 2.3), x, axis=0).astype(np.float32)
    x = signal.sosfilt(_peaking_eq(2100.0, 0.9, 1.4), x, axis=0).astype(np.float32)

    # 4. ВЧ-предыскажения -> асимметричное магнитное насыщение -> обратная коррекция
    sos_pre = _peaking_eq(3400.0, 0.7, 4.5)
    sos_de = _peaking_eq(3400.0, 0.7, -4.5)
    x_pre = signal.sosfilt(sos_pre, x, axis=0).astype(np.float32)
    drive, bias = 1.65, 0.045
    x_sat = (np.tanh(drive * x_pre + bias) - np.tanh(bias)) / np.tanh(drive)
    x = signal.sosfilt(sos_de, x_sat, axis=0).astype(np.float32)

    # 5. Модуляционная шероховатость магнитного слоя (asperity noise, 550–5200 Гц)
    sos_asp = signal.butter(2, [550.0 / (SR / 2.0), 5200.0 / (SR / 2.0)], btype='bandpass', output='sos')
    asp_noise = rng.standard_normal((n_samples, 2)).astype(np.float32)
    asp_noise = signal.sosfilt(sos_asp, asp_noise, axis=0).astype(np.float32)
    asp_noise /= (np.std(asp_noise) + 1e-6)
    x = x * (1.0 + 0.016 * asp_noise)

    # 6. Микро-колебания прижима ленты к головке
    drop_low = rng.standard_normal((n_samples // 200 + 10, 2)).astype(np.float32)
    sos_dl = signal.butter(2, 0.08, btype='lowpass', output='sos')
    drop_low = signal.sosfilt(sos_dl, drop_low, axis=0)
    drop_low /= (np.std(drop_low) + 1e-6)
    drop_env = 1.0 - np.clip(drop_low - 0.8, 0.0, 2.5) * 0.035
    x *= np.repeat(drop_env, 200, axis=0)[:n_samples]

    # 7. Деликатный аналоговый шум кассеты и микро-потрескивание оксида
    hiss_raw = rng.standard_normal((n_samples, 2)).astype(np.float32)
    sos_hiss = signal.butter(2, [300.0 / (SR / 2.0), 8200.0 / (SR / 2.0)], btype='bandpass', output='sos')
    sos_pink = signal.butter(1, 1400.0 / (SR / 2.0), btype='lowpass', output='sos')
    hiss = signal.sosfilt(sos_hiss, hiss_raw, axis=0).astype(np.float32)
    hiss = 0.65 * signal.sosfilt(sos_pink, hiss, axis=0).astype(np.float32) + 0.35 * hiss
    hiss /= (np.std(hiss) + 1e-6)

    ticks = (rng.random((n_samples, 2)) < 0.00018).astype(np.float32) * rng.standard_normal((n_samples, 2)).astype(np.float32)
    sos_tick = signal.butter(2, [600.0 / (SR / 2.0), 4800.0 / (SR / 2.0)], btype='bandpass', output='sos')
    ticks = signal.sosfilt(sos_tick, ticks, axis=0).astype(np.float32)

    sig_env = np.abs(x).mean(axis=1, keepdims=True)
    x = x + hiss * (0.0068 + 0.009 * sig_env) + ticks * 0.012

    # 8. Спад ВЧ магнитной головки и лёгкое взаимопроникновение стереоканалов
    sos_lp1 = signal.butter(2, 11200.0 / (SR / 2.0), btype='lowpass', output='sos')
    sos_lp2 = signal.butter(2, 13800.0 / (SR / 2.0), btype='lowpass', output='sos')
    x = signal.sosfilt(sos_lp1, x, axis=0).astype(np.float32)
    x = signal.sosfilt(sos_lp2, x, axis=0).astype(np.float32)

    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1]) * 0.84
    x[:, 0] = mid + side
    x[:, 1] = mid - side

    peak = float(np.max(np.abs(x)))
    if peak > 0.91:
        x *= (0.91 / peak)

    cmd_out = [
        ffmpeg, '-y', '-v', 'error',
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '2', '-ar', str(SR), '-i', '-',
        '-c:a', 'libmp3lame', '-b:a', '160k', str(OUT_MP3),
    ]
    proc = subprocess.Popen(cmd_out, stdin=subprocess.PIPE)
    proc.communicate(input=x.astype(np.float32).tobytes())
    if proc.returncode != 0:
        raise RuntimeError('Ошибка кодирования MP3 через ffmpeg')
    print(f'Кассетный трек: {OUT_MP3} ({OUT_MP3.stat().st_size} байт)')


if __name__ == '__main__':
    build_cassette_track()
