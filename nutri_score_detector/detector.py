"""Erkennung des Nutri-Score-Logos auf Verpackungsfronten.

Analyse eines Bildes (Foto oder Scan der Verpackungsvorderseite):
  1. Ist ein Nutri-Score vorhanden?
  2. Welcher Score (A-E)?
  3. Position (Bounding Box) und Groesse auf der Vorderseite?

Ansatz:
  1. HSV-Segmentierung der 5 offiziellen Nutri-Score-Farben (tolerant
     gegenueber Beleuchtung/Entsaettigung).
  2. Farbkomponenten in lesbare Gruppen clustern (Kreise/Kisten nahe beieinander).
  3. Verifikation ueber die offizielle Logo-Geometrie (aus dem SVG der
     Charta gemessen): 5 Farbbaender gleicher Hoehe in einer Reihe, das
     Score-Band als "Kapsel" deutlich breiter (1.2x) und hoeher (1.25x)
     als die uebrigen Baender, mit grossem weissen Buchstaben.
  4. Score = Farbe der Kapsel; Rueckfallebene: weisse Buchstaben-Lochflaechen.
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
        aspect_lo: float = 0.2,
        aspect_hi: float = 5.5,
        capsule_h_ratio: float = 1.10,
        capsule_w_ratio: float = 0.92,
        band_h_consistency: float = 0.72,
        min_hole_px: int = 12,
        hole_margin: float = 2.00,
        upscale_min_dim: int = 400,
    ) -> None:
        self.min_group_fraction = min_group_fraction
        self.max_group_fraction = max_group_fraction
        self.min_color_share = min_color_share
        self.min_colors = min_colors
        self.min_component_px = min_component_px
        self.aspect_lo = aspect_lo
        self.aspect_hi = aspect_hi
        self.capsule_h_ratio = capsule_h_ratio
        self.capsule_w_ratio = capsule_w_ratio
        self.band_h_consistency = band_h_consistency
        self.min_hole_px = min_hole_px
        self.hole_margin = hole_margin
        self.upscale_min_dim = upscale_min_dim

    def color_masks(self, hsv: np.ndarray) -> dict[str, np.ndarray]:
        masks = {}
        for letter, spec in COLORS.items():
            m = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in spec["hsv_ranges"]:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            masks[letter] = m
        return masks

    def _suppress_overlaps(self, entries: list[dict]) -> list[dict]:
        """Komponenten verschiedener Farben, deren Boxen sich stark ueberlappen,
        sind Segmentierungsartefakte (z.B. C-Band verschmolzen mit der D-Kapsel);
        nur die dominante bleibt."""
        keep = [True] * len(entries)
        for i, a in enumerate(entries):
            if not keep[i]:
                continue
            for j, b in enumerate(entries):
                if i == j or a["letter"] == b["letter"] or not keep[j]:
                    continue
                ix = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                iy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
                if ix <= 0 or iy <= 0:
                    continue
                if ix * iy >= 0.7 * min(a["area"], b["area"]):
                    if a["area"] < b["area"]:
                        keep[i] = False
                        break
                    keep[j] = False
        return [e for e, k in zip(entries, keep) if k]

    def _transform(self, boxes: dict, rot: int, W: int, H: int) -> dict[str, dict]:
        """Bilde Banderkisten in eine hanhaafte Ausrichtung (0/90/180/270)."""
        if rot == 0:
            return boxes
        out = {}
        for L, b in boxes.items():
            x, y, w, h = b["x"], b["y"], b["w"], b["h"]
            if rot == 90:
                nx, ny, nw, nh = y, W - (x + w), h, w
            elif rot == 180:
                nx, ny, nw, nh = W - (x + w), H - (y + h), w, h
            else:
                nx, ny, nw, nh = H - (y + h), x, h, w
            out[L] = {"x": nx, "y": ny, "w": nw, "h": nh, "area": b["area"]}
        return out

    def _staircase_ok(self, boxes: dict) -> tuple[bool, float]:
        """Treppenmuster A->E: x wandert nach rechts, hoehen plausibel,
        Obergrenze nicht voran (leichtes Absenken erlaubt, kein Sprung nach oben)."""
        keys = [L for L in SCORE_LETTERS if L in boxes]
        hs = [boxes[L]["h"] for L in keys]
        xs = [boxes[L]["x"] + boxes[L]["w"] / 2 for L in keys]
        tops = [boxes[L]["y"] for L in keys]
        hmax = max(hs)
        if len(hs) < self.min_colors:
            return False, 0.0
        for i in range(len(keys) - 1):
            if xs[i + 1] < xs[i] - 0.05 * hmax:
                return False, 0.0
            if tops[i + 1] < tops[i] - (0.5 * max(hs[i], 1) + 4):
                return False, 0.0
            if hs[i + 1] > 1.45 * hs[i] + 2:
                return False, 0.0
            if hs[i + 1] < 0.38 * hs[i] - 2:
                return False, 0.0
        return True, 1.0

    @staticmethod
    def _max_box_overlap(regions: dict) -> float:
        """Stark ueberlappende Paare von Farbkomponenten sind ein Hinweis auf
        verschmolzene Segmentierung (Artefakt); bestes Paar-Uberlapp-Mass."""
        vals = list(regions.values())
        worst = 0.0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, b = vals[i], vals[j]
                ix = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                iy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
                if ix <= 0 or iy <= 0:
                    continue
                worst = max(worst, min(1.0, ix * iy / min(a["area"], b["area"])))
        return worst

    def _capsule_letter(self, tb: dict, present: list[str]) -> tuple[str | None, dict]:
        """Offizielle Geometrie: das Score-Band ist Kapsel und damit messbar
        breiter und hoeher als die uebrigen Baender. Liefert (Farbe, Verhaeltnisse)."""
        hs = [tb[L]["h"] for L in present]
        ws = [tb[L]["w"] for L in present]
        h_med = float(np.median(hs))
        w_med = float(np.median(ws))
        if h_med <= 0 or w_med <= 0:
            return None, {}
        ratios = {L: round(min(tb[L]["h"] / h_med, tb[L]["w"] / w_med), 3)
                  for L in present}
        cap = max(ratios, key=ratios.get)
        if len(present) >= self.min_colors \
                and tb[cap]["h"] >= self.capsule_h_ratio * h_med \
                and tb[cap]["w"] >= self.capsule_w_ratio * w_med:
            # Offizielle Geometrie: alle NICHT-Score-Baender sind gleich hoch
            others = sorted(tb[L]["h"] for L in present if L != cap)
            o_med = float(np.median(others))
            if o_med > 0 and min(others) < self.band_h_consistency * o_med:
                return None, ratios
            return cap, ratios
        return None, ratios

    def analyze(self, image_bgr: np.ndarray) -> dict:
        h_img, w_img = image_bgr.shape[:2]
        img_area = h_img * w_img

        scale = 1
        if min(h_img, w_img) < self.upscale_min_dim:
            scale = -(-self.upscale_min_dim // max(min(h_img, w_img), 1))
            image_bgr = cv2.resize(image_bgr, (w_img * scale, h_img * scale),
                                   interpolation=cv2.INTER_CUBIC)
        r = self._analyze_scaled(image_bgr)
        if scale > 1 and r.get("bbox"):
            bx, by, bw, bh = (max(1, round(r["bbox"][k] / scale))
                              for k in ("x", "y", "w", "h"))
            r["bbox"] = {"x": bx, "y": by, "w": bw, "h": bh}
            r["bbox_relative"] = {"x": round(bx / w_img, 4), "y": round(by / h_img, 4),
                                  "w": round(bw / w_img, 4), "h": round(bh / h_img, 4)}
            r["size"] = {"width_px": bw, "height_px": bh,
                         "height_rel_to_image_height": round(bh / h_img, 4),
                         "area_fraction_of_image": round(bw * bh / img_area, 5)}
        return r

    def _analyze_scaled(self, image_bgr: np.ndarray) -> dict:
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

        results = self._anchor_candidates(entries, w_img, h_img)
        results += self._anchor_candidates(self._suppress_overlaps(entries), w_img, h_img)

        best = None
        for (regions, rot) in results:
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
            r_xy = bw / max(bh, 1)
            r_yx = bh / max(bw, 1)
            if not (min(r_xy, r_yx) >= self.aspect_lo and max(r_xy, r_yx) <= self.aspect_hi):
                continue
            aspect = r_xy
            coverage = total / (bw * bh)
            if coverage < 0.10:
                continue

            present = [L for L, v in counts.items() if v >= self.min_color_share * total]
            if len(present) < self.min_colors:
                continue

            tb = self._transform(regions, rot, w_img, h_img)
            capsules, ratios = self._capsule_letter(tb, present)
            holes = self._letter_holes(hsv, regions)
            hole_sum = sum(holes.values())
            ranked = sorted(holes.values(), reverse=True)
            letter_margin = ranked[0] / max(ranked[1], 1) if hole_sum > 0 else 0.0

            if capsules is not None:
                score_letter = capsules
                evidence = 2
            elif hole_sum >= self.min_hole_px and letter_margin >= self.hole_margin:
                score_letter = max(holes, key=holes.get)
                evidence = 1
            else:
                continue

            purity = 1.0 - self._max_box_overlap(regions)
            candidate = {
                "bbox": {"x": int(bb_x0), "y": int(bb_y0), "w": int(bw), "h": int(bh)},
                "counts": counts,
                "holes": holes,
                "capsule_ratios": ratios,
                "colors_present": sorted(present),
                "n_colors": len(present),
                "score": score_letter,
                "evidence": evidence,
                "purity": purity,
                "dominant_share": float(counts[score_letter] / total),
                "colored_share": float(total / (bw * bh)),
                "letter_margin": float(letter_margin),
                "step_score": step_score,
                "aspect": float(aspect),
                "rank": (evidence, purity, len(present), step_score, round(coverage, 2)),
            }
            if best is None or candidate["rank"] >= best["rank"]:
                best = candidate

        return self._build_result(best, img_area, w_img, h_img)

    def _anchor_candidates(self, entries: list[dict], W: int, H: int) -> list[tuple[dict, int]]:
        """Jede Komponente ist Anker; Kombinatorik ueber die K naechsten
        Komponenten je Farbe, 4 Rotationen. Ergebnis: (regionen, rotation)."""
        by_letter = {L: [e for e in entries if e["letter"] == L] for L in SCORE_LETTERS}
        topk = 4
        near: dict[int, dict[str, list[dict]]] = {}
        for idx, anchor in enumerate(entries):
            acx, acy = anchor["x"] + anchor["w"] / 2, anchor["y"] + anchor["h"] / 2
            reach = 4.0 * max(anchor["w"], anchor["h"]) + 20
            near[idx] = {}
            for L in SCORE_LETTERS:
                if L == anchor["letter"]:
                    continue
                pool = []
                for e in by_letter.get(L, []):
                    dx = e["x"] + e["w"] / 2 - acx
                    dy = e["y"] + e["h"] / 2 - acy
                    d = (dx * dx + dy * dy) ** 0.5
                    if d <= reach:
                        pool.append((d, e))
                pool.sort(key=lambda t: t[0])
                near[idx][L] = [e for _d, e in pool[:topk]]

        seen = set()
        candidates = []
        sub_sets = [None] + [[L] for L in SCORE_LETTERS]
        for idx, anchor in enumerate(entries):
            for omit in sub_sets:
                pools = {}
                n_opts = 1
                for L, pool in near[idx].items():
                    if omit and L in omit:
                        continue
                    pools[L] = pool
                    n_opts *= len(pool)
                for combo_i in range(n_opts):
                    regions = {anchor["letter"]: anchor}
                    tmp = combo_i
                    for L, pool in pools.items():
                        regions[L] = pool[tmp % len(pool)]
                        tmp //= len(pool)
                    if len(regions) < self.min_colors:
                        continue
                    key = tuple(sorted((L, r["x"], r["y"], r["w"], r["h"])
                                       for L, r in regions.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    for rot in (0, 90, 180, 270):
                        tb = self._transform(regions, rot, W, H)
                        ok, step = self._staircase_ok(tb)
                        if ok:
                            candidates.append((regions, rot))
                            break
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
        if best["evidence"] >= 2:
            return round(min(1.0, 0.60 + 0.08 * best["n_colors"]
                             + 0.05 * best["rank"][2] + 0.05 * best["purity"]), 3)
        return round(0.35 + 0.10 * best["n_colors"], 3)


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
