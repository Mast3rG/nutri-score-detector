"""Evaluation des Detektors auf synthetischen Verpackungsfronten."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2

from .colors import SCORE_LETTERS
from .detector import NutriScoreDetector, analyze_file, annotate
from .make_test_images import synth_package


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def evaluate(count: int = 120, seed: int = 42) -> dict:
    rng = random.Random(seed)
    det = NutriScoreDetector()
    presence_ok = 0
    n_with = found = score_ok = 0
    iou_sum = 0.0
    iou_n = 0

    for _ in range(count):
        w = rng.choice([900, 1024, 1200])
        h = rng.choice([1200, 1350, 1500])
        score = None if rng.random() < 0.25 else rng.choice(SCORE_LETTERS)
        img, gt = synth_package(w, h, score, rng)
        res = det.analyze(img)
        if (res["nutri_score_present"]) == (score is not None):
            presence_ok += 1
        if score is not None:
            n_with += 1
            if res["nutri_score_present"]:
                found += 1
                if res["score"] == score:
                    score_ok += 1
                b = res["bbox"]
                v = iou((b["x"], b["y"], b["w"], b["h"]), gt)
                iou_sum += v
                iou_n += 1

    return {
        "images": count,
        "presence_accuracy": round(presence_ok / count, 3),
        "logo_found_share": round(found / max(n_with, 1), 3),
        "score_accuracy_when_found": round(score_ok / max(found, 1), 3),
        "mean_iou_when_found": round(iou_sum / max(iou_n, 1), 3),
    }


def evaluate_benchmark(path: str = "testdata/benchmark.json") -> dict:
    """Bewertet den Detektor gegen echte Fotos mit eingetragener Wahrheit.

    Erwartet eine JSON-Datei mit images: [{file, score (str|list|null),
    score_unsicher, bemerkung}, ...]. score=null = kein Logo.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    present_ok = ok = n = n_certain = ok_certain = 0
    details = []
    for entry in data.get("images", []):
        file = entry["file"]
        truth = entry["score"]
        accepted = truth if isinstance(truth, list) else ([truth] if truth is not None else [])
        try:
            res = analyze_file(file)
        except (ValueError, OSError) as exc:
            details.append({"file": file, "error": str(exc)})
            continue
        got = res["score"] if res.get("nutri_score_present") else None
        item = {"file": file, "erwartet": truth, "erhalten": got}
        n += 1
        if (got is None) == (truth is None):
            present_ok += 1
        if truth is not None:
            n_certain += 0 if entry.get("score_unsicher") else 1
            if got in accepted:
                ok += 1
                ok_certain += 0 if entry.get("score_unsicher") else 1
        details.append(item)

    return {
        "images": n,
        "presence_accuracy": round(present_ok / max(n, 1), 3),
        "score_accuracy": round(ok / max(n, 1), 3),
        "score_accuracy_certain_only": round(ok_certain / max(n_certain, 1), 3),
        "nicht_bewertbare_score_faelle": n - n_certain,
        "details": details,
    }


def save_demo(out_dir: str = "testdata", count: int = 15, seed: int = 42) -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    rng = random.Random(seed)
    det = NutriScoreDetector()
    for i in range(count):
        score = None if i % 5 == 4 else SCORE_LETTERS[i % 5]
        img, _gt = synth_package(1024, 1440, score, rng)
        res = det.analyze(img)
        cv2.imwrite(str(out / f"demo_{i:02d}_src.png"), img)
        cv2.imwrite(str(out / f"demo_{i:02d}_detected_{(res['score'] or 'none')}.png"),
                    annotate(img, res))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(json.dumps(evaluate_benchmark(sys.argv[1]), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(evaluate(), indent=2))
