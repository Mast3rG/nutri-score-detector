"""Erkennung des Nutri-Score-Logos auf Verpackungsfronten.

Analyse eines Bildes (Foto oder Scan der Verpackungsvorderseite):
  1. Ist ein Nutri-Score vorhanden?
  2. Welcher Score (A-E)?
  3. Position (Bounding Box) und Groesse auf der Vorderseite?

Ansatz:
  1. HSV-Segmentierung der 5 offiziellen Nutri-Score-Farben (tolerant
     gegenueber Beleuchtung/Entsaettigung).
  2. Farbkomponenten in lesbare Gruppen clustern (Kreise/Kisten nahe beieinander).
  3. Verifikation ueber das charakteristische Treppenmuster: 5 Farbbalken
     nebeneinander, von A (oben links) nach E (unten rechts) abfallend,
     Balken rechts/unter dem Score kleiner als davor.
  4. Score = letzte "vollhoehe" Farbbahn; staerkt durch Vergleich der
     weissen Buchstaben-Lochflaechen.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colors import COLORS, SCORE_LETTERS


def load_image(path: str) -> np.ndarray | None:
    """Unicode-sicheres Laden (BGR)."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class NutriScoreDetector:
    def __init__(
        self,
        min_group_fraction: float = 0.0006,
        max_group_fraction: float = 0.85,
        min_color_share: float = 0.02,
        min_colors: int = 4,
        min_component_px: int = 25,
        aspect_lo: float = 0.45,
        aspect_hi: float = 2.8,
        expand_frac: float = 0.45,
        height_rise: float = 1.12,
        height_drop: float = 0.30,
        full_height_frac: float = 0.62,
    ) -> None:
        self.min_group_fraction = min_group_fraction
        self.max_group_fraction = max_group_fraction
        self.min_color_share = min_color_share
        self.min_colors = min_colors
        self.min_component_px = min_component_px
        self.aspect_lo = aspect_lo
        self.aspect_hi = aspect_hi
        self.expand_frac = expand_frac
        self.height_rise = height_rise
        self.height_drop = height_drop
        self.full_height_frac = full_height_frac

    def color_masks(self, hsv: np.ndarray) -> dict[str, np.ndarray]:
        masks = {}
        for letter, spec in COLORS.items():
            m = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in spec["hsv_ranges"]:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            masks[letter] = m
        return masks

    def _per_color_regions(self, entries: list[dict], bbox) -> dict[str, dict]:
        """Groesste Farbkomponente je Farbe innerhalb der BBox."""
        x0, y0, x1, y1 = bbox
        out = {}
        for letter in SCORE_LETTERS:
            cands = [e for e in entries
                     if e["letter"] == letter and e["w"] <= (x1 - x0) * 1.05
                     and x0 - 4 <= (e["x"] + e["w"] / 2) <= x1 + 4
                     and y0 - 4 <= (e["y"] + e["h"] / 2) <= y1 + 4]
            if cands:
                out[letter] = max(cands, key=lambda e: e["area"])
        return out

    def _staircase_ok(self, boxes: dict[str, dict]) -> tuple[bool, float]:
        """Treppenmuster A->E: x wandert nach rechts, Kanten sinken nie stark,
         Hoehen bleiben plausibel (kein Anstieg >1.12x, kein Einbruch <0.30x)."""
        keys = [L for L in SCORE_LETTERS if L in boxes]
        hs = [boxes[L]["h"] for L in keys]
        xs = [boxes[L]["x"] + boxes[L]["w"] / 2 for L in keys]
        tops = [boxes[L]["y"] for L in keys]
        hmax = max(hs)
        for i in range(len(keys) - 1):
            if xs[i + 1] < xs[i] - 0.05 * hmax:
                return False, 0.0
            if tops[i + 1] < tops[i] - 0.35 * max(hs[i], 1):
                return False, 0.0
            if hs[i + 1] > self.height_rise * hs[i] + 2:
                return False, 0.0
            if hs[i + 1] < self.height_drop * hs[i] - 2:
                return False, 0.0
        return True, 1.0

    def analyze(self, image_bgr: np.ndarray) -> dict:
        h_img, w_img = image_bgr.shape[:2]
        img_area = h_img * w_img
        min_px = max(self.min_component_px, int(img_area * 0.00001))

        blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        masks = self.color_masks(hsv)

        entries = []
        for letter, m in masks.items():
            n, _lab, stats, _ = cv2.connectedComponentsWithStats(m)
            for c in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[c])
                if area < min_px or w >= w_img or h >= h_img:
                    continue
                entries.append({"letter": letter, "x": x, "y": y,
                                "w": w, "h": h, "area": area})

        results = self._anchor_candidates(entries)

        best = None
        for regions in results:
            if len(regions) < self.min_colors:
                continue
            ok_step, step_score = self._staircase_ok(regions)
            if not ok_step:
                continue

            bb_x0 = min(regions[L]["x"] for L in regions)
            bb_y0 = min(regions[L]["y"] for L in regions)
            bb_x1 = max(regions[L]["x"] + regions[L]["w"] for L in regions)
            bb_y1 = max(regions[L]["y"] + regions[L]["h"] for L in regions)
            bw, bh = bb_x1 - bb_x0, bb_y1 - bb_y0

            counts = {L: r["area"] for L, r in regions.items()}
            total = sum(counts.values())
            if total == 0:
                continue
            frac = (bw * bh) / img_area
            if not (self.min_group_fraction <= frac <= self.max_group_fraction):
                continue
            aspect = bw / max(bh, 1)
            if not (self.aspect_lo <= aspect <= self.aspect_hi):
                continue
            coverage = total / (bw * bh)
            if coverage < 0.10:
                continue

            present = [L for L, v in counts.items() if v >= self.min_color_share * total]
            if len(present) < self.min_colors:
                continue

            hs = [regions[L]["h"] for L in SCORE_LETTERS if L in regions]
            full = [L for L in SCORE_LETTERS
                    if L in regions and regions[L]["h"] >= self.full_height_frac * max(hs)]
            holes = self._letter_holes(hsv, regions)
            score_letter = full[-1] if full else max(counts, key=counts.get)
            hole_sum = sum(holes.values())
            ranked = sorted(holes.values(), reverse=True)
            letter_margin = ranked[0] / max(ranked[1], 1) if hole_sum > 0 else 0.0
            holes_max = max(holes, key=holes.get) if hole_sum > 0 else None
            if (holes_max is not None and letter_margin >= 1.6
                    and full and holes_max in full and holes_max != score_letter
                    and SCORE_LETTERS.index(score_letter) < SCORE_LETTERS.index(holes_max)):
                score_letter = holes_max

            dominant_share = counts[score_letter] / total
            confidence = round(min(1.0, 0.25 * len(present) / 5
                                   + 0.45 * step_score
                                   + 0.20 * min(letter_margin, 2.0) / 2.0
                                   + 0.10), 3)

            candidate = {
                "bbox": {"x": int(bb_x0), "y": int(bb_y0), "w": int(bw), "h": int(bh)},
                "counts": counts,
                "holes": holes,
                "colors_present": sorted(present),
                "n_colors": len(present),
                "score": score_letter,
                "dominant_share": float(dominant_share),
                "colored_share": float(total / (bw * bh)),
                "letter_margin": float(letter_margin),
                "step_score": step_score,
                "aspect": float(aspect),
                "rank": (len(present), step_score, round(coverage, 3)),
            }
            if best is None or candidate["rank"] >= best["rank"]:
                best = candidate

        return self._build_result(best, img_area, w_img, h_img)

    def _anchor_candidates(self, entries: list[dict]) -> list[dict]:
        """Jede Komponente ist Anker; um sie werden die naechsten Komponenten
        der anderen Farben gesucht -> Kandidat aus 5 (bzw. 4) Baendern."""
        by_letter = {L: [e for e in entries if e["letter"] == L] for L in SCORE_LETTERS}
        candidates = []
        for anchor in entries:
            regions = {anchor["letter"]: anchor}
            acx, acy = anchor["x"] + anchor["w"] / 2, anchor["y"] + anchor["h"] / 2
            reach = 3.5 * max(anchor["w"], anchor["h"]) + 20
            ok = True
            for L in SCORE_LETTERS:
                if L == anchor["letter"]:
                    continue
                cands = by_letter.get(L, [])
                if not cands:
                    ok = False
                    break
                nb = min(cands, key=lambda e: (e["x"] + e["w"] / 2 - acx) ** 2
                                           + (e["y"] + e["h"] / 2 - acy) ** 2)
                dx = nb["x"] + nb["w"] / 2 - acx
                dy = nb["y"] + nb["h"] / 2 - acy
                if (dx * dx + dy * dy) ** 0.5 > reach:
                    ok = False
                    break
                regions[L] = nb
            if ok and len(regions) >= self.min_colors:
                candidates.append(regions)
        return candidates

    def _letter_holes(self, hsv, regions) -> dict[str, int]:
        """Weisse Buchstabenflaeche je Farbband (Loch im groessten Farbbaulk)."""
        pads = 1
        holes = {}
        for letter, r in regions.items():
            bx0 = max(0, r["x"] - pads); by0 = max(0, r["y"] - pads)
            bx1 = min(hsv.shape[1], r["x"] + r["w"] + pads)
            by1 = min(hsv.shape[0], r["y"] + r["h"] + pads)
            band = COLORS[letter]
            m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
            for lo, hi in band["hsv_ranges"]:
                m |= cv2.inRange(hsv[by0:by1, bx0:bx1],
                                 np.array(lo, np.uint8), np.array(hi, np.uint8))
            nb, labb, statb, _ = cv2.connectedComponentsWithStats(m)
            if nb < 2:
                holes[letter] = 0
                continue
            li = 1 + int(np.argmax(statb[1:, 4]))
            b = (labb == li).astype(np.uint8)
            contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(b)
            cv2.drawContours(filled, contours, -1, 1, -1)
            holes[letter] = int(np.count_nonzero(filled - b))
        return holes

    def _build_result(self, best, img_area, w_img, h_img) -> dict:
        if best is None:
            return {
                "nutri_score_present": False,
                "score": None,
                "confidence": 0.0,
                "bbox": None,
                "bbox_relative": None,
                "size": None,
            }

        x, y, w, h = (best["bbox"][k] for k in ("x", "y", "w", "h"))
        return {
            "nutri_score_present": True,
            "score": best["score"],
            "confidence": self._confidence(best),
            "bbox": best["bbox"],
            "bbox_relative": {
                "x": round(x / w_img, 4),
                "y": round(y / h_img, 4),
                "w": round(w / w_img, 4),
                "h": round(h / h_img, 4),
            },
            "size": {
                "width_px": int(w),
                "height_px": int(h),
                "height_rel_to_image_height": round(h / h_img, 4),
                "area_fraction_of_image": round((w * h) / img_area, 5),
            },
            "aspect_ratio": round(best["aspect"], 3),
            "color_shares": {k: round(v / sum(best["counts"].values()), 3)
                             for k, v in best["counts"].items()},
            "letter_pixels": best["holes"],
            "colors_detected": best["colors_present"],
        }

    def _confidence(self, best) -> float:
        return round(min(1.0, 0.25 * best["n_colors"] / 5
                         + 0.45 * best["rank"][1]
                         + 0.20 * min(best["letter_margin"], 2.0) / 2.0
                         + 0.10), 3)


def annotate(image_bgr: np.ndarray, result: dict) -> np.ndarray:
    """Zeichnet Bounding Box + Score zum visuellen Check."""
    out = image_bgr.copy()
    if result.get("nutri_score_present") and result.get("bbox"):
        x, y, w, h = (result["bbox"][k] for k in ("x", "y", "w", "h"))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 6)
        text = f"{result['score']} ({result['confidence']:.2f})"
        cv2.putText(out, text, (x, max(28, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def analyze_file(path: str, debug_out: str | None = None) -> dict:
    image = load_image(path)
    if image is None:
        raise ValueError(f"Bild konnte nicht gelesen werden: {path}")
    detector = NutriScoreDetector()
    result = detector.analyze(image)
    result["file"] = path
    if debug_out:
        canvas = annotate(image, result)
        ok, buf = cv2.imencode(".png", canvas)
        if ok:
            buf.tofile(debug_out)
    return result


def format_result(result: dict) -> str:
    lines = [f"Datei: {result.get('file', '-')}"]
    if not result.get("nutri_score_present"):
        lines.append("Nutri-Score: NICHT gefunden")
        return "\n".join(lines)
    lines.append(f"Nutri-Score: VORHANDEN -> {result['score']} (Konfidenz {result['confidence']})")
    b = result["bbox"]
    lines.append(f"Position: x={b['x']}, y={b['y']}, Breite={b['w']} px, Hoehe={b['h']} px")
    r = result["bbox_relative"]
    lines.append(
        f"Relativ: links {r['x']:.1%}, oben {r['y']:.1%}, "
        f"Breite {r['w']:.1%} der Bildbreite, Hoehe {r['h']:.1%} der Bildhoehe"
    )
    return "\n".join(lines)
