"""Erzeugt synthetische Verpackungsfronten mit Nutri-Score-Logo zum Testen.

Das Logo wird schematisch gezeichnet: 5 Farbbalken (A-E) mit weissen
Buchstaben, duennen weissen Trennfugen, der aktive Abschnitt vergroessert
(grosser Buchstabe), Abschnitte nach dem Score verkleinert - wie im
Original-Logo. Rueckgabe enthaelt die Ground-Truth-Bounding-Box.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .colors import COLORS, SCORE_LETTERS

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_logo(canvas, x0, y0, height, score):
    h_small = max(int(height * 0.45), 4)
    band_w_big = max(int(round(height / 5 * 1.15)), 6)
    band_w_small = max(int(round(band_w_big * 0.45)), 3)
    gap = max(height // 60, 2)
    n_full = SCORE_LETTERS.index(score) + 1

    widths = [band_w_big if i < n_full else band_w_small for i in range(5)]
    for i, letter in enumerate(SCORE_LETTERS):
        w = widths[i]
        left = x0 + sum(widths[:i]) + i * gap
        top = y0 if i < n_full else y0 + height - h_small
        color = COLORS[letter]["rgb"][::-1]
        cv2.rectangle(canvas, (left, top), (left + w, y0 + height), color, -1)

        fs = 2.6 if letter == score else 0.9
        (tw, th), _ = cv2.getTextSize(letter, FONT, fs, 3)
        fs = min(fs, fs * (0.85 * w) / max(tw, 1), fs * (0.6 * (y0 + height - top)) / max(th, 1))
        thickness = max(int(round(2.5 * fs)), 2)
        (tw, th), _ = cv2.getTextSize(letter, FONT, fs, thickness)
        cx, cy = left + w // 2, (top + y0 + height) // 2
        cv2.putText(canvas, letter, (cx - tw // 2, cy + th // 2), FONT, fs,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    total_w = sum(widths) + 4 * gap
    return x0, y0, total_w, height


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
        logo_h = rng.randint(height // 14, height // 4)
        approx_w = int(logo_h * 1.4)
        x0 = rng.randint(pad, max(pad + 1, width - approx_w - pad))
        y0 = rng.randint(pad, max(pad + 1, height - logo_h - pad))
        gt = draw_logo(img, x0, y0, logo_h, score)

    for _ in range(rng.randint(0, 3)):
        letter = rng.choice(SCORE_LETTERS)
        color = COLORS[letter]["rgb"][::-1]
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
