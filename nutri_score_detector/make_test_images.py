"""Erzeugt synthetische Verpackungsfronten mit Nutri-Score-Logo zum Testen.

Das Logo entspricht der offiziellen Geometrie (aus der Charta-SVG gemessen):
5 Farbbaender gleicher Hoehe in einer Reihe, das Score-Band als breitere,
hoechere "Kapsel" mit grossem weissen Buchstaben, die uebrigen Baender mit
kleinen weissen Buchstaben, graues "NUTRI-SCORE"-Wort darueber. Rueckgabe
enthaelt die Ground-Truth-Bounding-Box der farbigen Flaeche.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .colors import COLORS, SCORE_LETTERS

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Verhaeltnisse aus dem offiziellen Logo (Bandreihe = 27.91 Einheiten)
ROW_H = 27.91
BAND_W = 17.15
ADJ_W = 14.90
CAP_W = 20.58
CAP_H = 34.89
LETTER_SMALL = 12.44
LETTER_BIG = 16.59


def _band(canvas, x0, y0, w, h, bgr, round_left=False, round_right=False):
    r = min(w, h) // 2
    x1 = x0 + w
    y1 = y0 + h
    xl = x0 + (r if round_left else 0)
    xr = x1 - (r if round_right else 0)
    if xr > xl:
        cv2.rectangle(canvas, (xl, y0), (xr, y1), bgr, -1)
    if round_left:
        cv2.ellipse(canvas, (x0 + r, y0 + h // 2), (r, h // 2), 0, 90, 270, bgr, -1)
    if round_right:
        cv2.ellipse(canvas, (x1 - r, y0 + h // 2), (r, h // 2), 0, -90, 90, bgr, -1)


def _put_letter(canvas, cx, cy, letter, target_h):
    fs, th = float(target_h), 2
    for _ in range(8):
        (tw, th), _ = cv2.getTextSize(letter, FONT, fs, th)
        if max(tw, th) <= 0:
            break
        if th <= target_h:
            break
        fs *= 0.9 * min(target_h / th, target_h / max(tw, 1))
    (tw, th), _ = cv2.getTextSize(letter, FONT, fs, 1)
    thickness = max(int(round(fs * 2.2)), 1)
    cv2.putText(canvas, letter, (cx - tw // 2, cy + th // 2), FONT, fs,
                (255, 255, 255), thickness, cv2.LINE_AA)


def _put_gray_word(canvas, x0, y0, text_h):
    word = "NUTRI-SCORE"
    fs, th = float(text_h), 1
    for _ in range(8):
        (tw, th), _ = cv2.getTextSize(word, FONT, fs, th)
        if max(tw, th) <= 0:
            break
        if th <= text_h:
            break
        fs *= 0.9 * min(text_h / th, text_h / max(tw, 1))
    (tw, th), _ = cv2.getTextSize(word, FONT, fs, 1)
    cv2.putText(canvas, word, (x0, y0 + th), FONT, fs,
                (109, 108, 111), max(int(round(fs * 1.6)), 1), cv2.LINE_AA)


def draw_logo(canvas, x0, y0, height, score):
    """Zeichnet das offizielle Logo; height = Hoehe der Bandreihe."""
    s = height / ROW_H
    gap = max(int(round(0.6 * s)), 1)

    idx = SCORE_LETTERS.index(score)
    widths = [ADJ_W if abs(i - idx) == 1 else BAND_W for i in range(5)]
    widths[idx] = CAP_W

    cap_x0 = int(round(x0 + sum(widths[:idx]) * s + idx * gap))
    cap_y0 = int(round(max(0, y0 - 0.125 * height)))
    cap_h = int(round(CAP_H * s))
    cap_w = int(round(CAP_W * s))

    bill = {}
    b_x = x0
    for i, letter in enumerate(SCORE_LETTERS):
        if i == idx:
            b_x += int(round(CAP_W * s)) + gap
            continue
        w = int(round(widths[i] * s))
        _band(canvas, b_x, y0, w, height, COLORS[letter]["rgb"][::-1],
              round_left=(i == 0), round_right=(i == 4))
        bill[letter] = (b_x, y0, w, height)
        b_x += w + gap

    _band(canvas, cap_x0, cap_y0, cap_w, cap_h, COLORS[score]["rgb"][::-1],
          round_left=True, round_right=True)

    for i, letter in enumerate(SCORE_LETTERS):
        if i == idx:
            _put_letter(canvas, cap_x0 + cap_w // 2, cap_y0 + cap_h // 2,
                        letter, LETTER_BIG * s)
        else:
            bx, by, ww, hh = bill[letter]
            _put_letter(canvas, bx + ww // 2, by + hh // 2,
                        letter, LETTER_SMALL * s)
    text_h = int(round(6.7 * s))
    if text_h >= 6:
        _put_gray_word(canvas, x0, max(0, cap_y0 - int(round(5.8 * s)) - text_h),
                       text_h)

    total_w = sum(int(round(w * s)) for w in widths) + 4 * gap
    return x0, cap_y0, total_w, cap_h


def synth_package(width, height, score, rng):
    """Liefert (Bild, Ground-Truth-BBox oder None)."""
    img = np.full((height, width, 3), (245, 240, 230), np.uint8)

    for _ in range(rng.randint(2, 5)):
        rw, rh = rng.randint(60, width // 4), rng.randint(20, height // 8)
        rx, ry = rng.randint(0, width - rw), rng.randint(0, height - rh)
        col = tuple(int(rng.randint(120, 230)) for _ in range(3))
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), col, -1)
    for _ in range(rng.randint(1, 3)):
        cv2.rectangle(img, (rng.randint(0, width - 10), rng.randint(0, height - 10)),
                      (rng.randint(0, width), rng.randint(0, height)), (60, 60, 60),
                      rng.randint(1, 3))

    gt = None
    if score is not None:
        pad = 30
        logo_h = rng.randint(max(height // 14, 24), height // 4)
        approx_w = int(logo_h * 2.9)
        x0 = rng.randint(pad, max(pad + 1, width - approx_w - pad))
        y0 = rng.randint(pad + logo_h // 8,
                         max(pad + logo_h // 8 + 1, height - logo_h - pad))
        gt = draw_logo(img, x0, y0, logo_h, score)

    k = rng.randint(1, 3) if score is not None else rng.randint(2, 4)
    for _ in range(k):
        letter = rng.choice(SCORE_LETTERS)
        color = COLORS[letter]["rgb"][::-1]
        rx = ry = rw = rh = 0
        for _try in range(20):
            rw, rh = rng.randint(40, width // 6), rng.randint(15, height // 12)
            rx, ry = rng.randint(0, width - rw), rng.randint(0, height - rh)
            if gt is None or rx > gt[0] + gt[2] + 20 or rx + rw < gt[0] - 20 \
               or ry > gt[1] + gt[3] + 20 or ry + rh < gt[1] - 20:
                break
        if rng.random() < 0.5:
            cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), color, -1)
        else:
            cv2.ellipse(img, (rx + rw // 2, ry + rh // 2), (rw // 2, rh // 2),
                        rng.randint(0, 180), 0, 360, color, -1)
    return img, gt


def main(count: int = 60, seed: int = 7, out_dir: str = "testdata") -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(count):
        w = rng.choice([900, 1024, 1200, 1440])
        h = rng.choice([1200, 1350, 1500, 1600])
        score = None if rng.random() < 0.25 else rng.choice(SCORE_LETTERS)
        img, _gt = synth_package(w, h, score, rng)
        tag = score if score else "none"
        cv2.imwrite(str(out / f"pkg_{i:03d}_{tag}.png"), img)
    print(f"{count} Bilder erzeugt in {out.resolve()}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(n)
