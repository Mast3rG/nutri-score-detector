"""Erkennung des Nutri-Score-Logos auf Verpackungsfronten.

Analyse eines Bildes (Foto oder Scan der Verpackungsvorderseite):
  1. Ist ein Nutri-Score vorhanden?
  2. Welcher Score (A-E)?
  3. Position (Bounding Box) und Groesse auf der Vorderseite?

Ansatz: Farbraum-Segmentierung in den 5 offiziellen Nutri-Score-Farben,
Zusammenfuehren zu Kandidatenregionen und Verifikation ueber das
farbliche Muster (alle 5 oder mind. 4 der 5 Farben muessen nennenswert
vertreten sein). Der Score ergibt sich aus der Farbe mit dem groessten
Flaechenanteil, da im Logo der aktive Buchstabe/abschnitt vergroessert
dargestellt wird.
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
        min_region_fraction: float = 0.0015,
        max_region_fraction: float = 0.9,
        min_color_share: float = 0.05,
        min_colors: int = 4,
        min_colored_area_share: float = 0.45,
        aspect_lo: float = 0.45,
        aspect_hi: float = 2.2,
    ) -> None:
        self.min_region_fraction = min_region_fraction
        self.max_region_fraction = max_region_fraction
        self.min_color_share = min_color_share
        self.min_colors = min_colors
        self.min_colored_area_share = min_colored_area_share
        self.aspect_lo = aspect_lo
        self.aspect_hi = aspect_hi

    def color_masks(self, hsv: np.ndarray) -> dict[str, np.ndarray]:
        masks = {}
        for letter, spec in COLORS.items():
            m = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in spec["hsv_ranges"]:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            masks[letter] = m
        return masks

    def _candidate_mask(self, combined: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
        closed = cv2.dilate(closed, k, iterations=1)
        return closed

    def analyze(self, image_bgr: np.ndarray) -> dict:
        h_img, w_img = image_bgr.shape[:2]
        img_area = h_img * w_img

        blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        masks = self.color_masks(hsv)
        combined = np.zeros_like(masks["A"])
        for m in masks.values():
            combined |= m

        closed = self._candidate_mask(combined)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

        best = None
        for label in range(1, n_labels):
            x, y, w, h, _area = stats[label]
            if w >= w_img or h >= h_img:
                continue
            region_area = float(w * h)
            fraction = region_area / img_area
            if not (self.min_region_fraction <= fraction <= self.max_region_fraction):
                continue
            aspect = w / max(h, 1)
            if not (self.aspect_lo <= aspect <= self.aspect_hi):
                continue

            comp = (labels[y:y + h, x:x + w] == label)
            counts = {}
            for letter, m in masks.items():
                region = m[y:y + h, x:x + w] & comp
                counts[letter] = int(np.count_nonzero(region))
            total = sum(counts.values())
            if total == 0:
                continue
            colored_share = total / region_area
            if colored_share < self.min_colored_area_share:
                continue

            shares = {k: v / total for k, v in counts.items()}
            present = [k for k, v in counts.items() if v >= self.min_color_share * max(counts.values())]
            if len(present) < self.min_colors:
                continue
            white_counts = {}
            band_heights = {}
            for letter, m in masks.items():
                band_local = ((m[y:y + h, x:x + w] > 0) & comp).astype(np.uint8)
                n_c, lab_c, stat_c, _ = cv2.connectedComponentsWithStats(band_local)
                if n_c < 2:
                    white_counts[letter] = 0
                    band_heights[letter] = 0
                    continue
                li = 1 + int(np.argmax(stat_c[1:, 4]))
                band = (lab_c == li).astype(np.uint8)
                contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(band)
                cv2.drawContours(filled, contours, -1, 1, -1)
                white_counts[letter] = int(np.count_nonzero(filled - band))
                band_heights[letter] = int(stat_c[li, 3])

            hole_sum = sum(white_counts.values())
            holes_letter = max(white_counts, key=white_counts.get)
            ranked = sorted(white_counts.values(), reverse=True)
            letter_margin = (ranked[0] / max(ranked[1], 1) if hole_sum > 0 else 0.0)
            full = [k for k in SCORE_LETTERS
                    if band_heights.get(k, 0) >= 0.7 * h and counts[k] > 0]
            score_letter = holes_letter
            if hole_sum == 0 or letter_margin < 1.25:
                if full:
                    score_letter = full[-1]
                elif hole_sum == 0:
                    score_letter = max(counts, key=counts.get)
                else:
                    score_letter = holes_letter
            dominant_share = counts[score_letter] / total
            candidate = {
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "counts": counts,
                "white_counts": white_counts,
                "shares": shares,
                "colors_present": sorted(present),
                "n_colors": len(present),
                "score": score_letter,
                "dominant_share": float(dominant_share),
                "colored_share": float(colored_share),
                "letter_margin": float(letter_margin),
                "aspect": float(aspect),
                "rank": (len(present), total),
            }
            if best is None or candidate["rank"] > best["rank"]:
                best = candidate

        return self._build_result(best, img_area, w_img, h_img)

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

        x, y, w, h = best["bbox"].values()
        second = sorted(best["shares"].values())[-2] if len(best["shares"]) > 1 else 0.0
        margin = best["dominant_share"] - second
        confidence = round(
            min(
                1.0,
                0.25 * best["n_colors"] / 5
                + 0.25 * min(best["letter_margin"], 2.0) / 2.0
                + 0.55 * best["dominant_share"]
                + 0.20 * best["colored_share"],
            ),
            3,
        )

        return {
            "nutri_score_present": True,
            "score": best["score"],
            "confidence": confidence,
            "margin_over_next_color": round(margin, 3),
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
                "area_fraction_of_image": best["bbox"].get("area", None)
                or round((w * h) / img_area, 5),
            },
            "aspect_ratio": round(best["aspect"], 3),
            "color_shares": {k: round(v, 3) for k, v in best["shares"].items()},
            "letter_pixels": best["white_counts"],
            "colors_detected": best["colors_present"],
        }


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
