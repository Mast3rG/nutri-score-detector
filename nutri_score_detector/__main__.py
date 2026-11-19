"""Kommandozeile: Nutri-Score auf Bildern pruefen.

Einzelbild:
    python -m nutri_score_detector pfad/zu/verpackung.jpg --debug debug.png
    python -m nutri_score_detector verpackung.jpg --json

Ganzer Ordner (Batch):
    python -m nutri_score_detector fotos/ --csv ergebnis.csv
    python -m nutri_score_detector fotos/ --csv out.csv --annotiert annotated/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .detector import analyze_file

BILD_ENDUNGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CSV_FELDER = [
    "datei", "nutri_score_present", "score", "confidence",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "rel_x", "rel_y", "rel_w", "rel_h",
    "area_fraction_of_image",
]


def result_row(path: str, result: dict) -> dict:
    row = {
        "datei": path,
        "nutri_score_present": int(bool(result.get("nutri_score_present"))),
        "score": result.get("score") or "",
        "confidence": result.get("confidence", 0.0),
    }
    bbox = result.get("bbox") or {}
    rel = result.get("bbox_relative") or {}
    size = result.get("size") or {}
    row.update({
        "bbox_x": bbox.get("x", ""), "bbox_y": bbox.get("y", ""),
        "bbox_w": bbox.get("w", ""), "bbox_h": bbox.get("h", ""),
        "rel_x": rel.get("x", ""), "rel_y": rel.get("y", ""),
        "rel_w": rel.get("w", ""), "rel_h": rel.get("h", ""),
        "area_fraction_of_image": size.get("area_fraction_of_image", ""),
    })
    return row


def run_batch(src: Path, csv_out: Path | None, out_dir: Path | None,
              jsonl_out: Path | None) -> int:
    if src.is_file():
        paths = [src]
    else:
        paths = sorted(p for p in src.iterdir()
                       if p.suffix.lower() in BILD_ENDUNGEN)
    if not paths:
        print(f"Keine Bilder gefunden in: {src}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    gefunden = 0
    errors = 0
    for p in paths:
        try:
            debug = None
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                debug = str(out_dir / f"{p.stem}_annotiert.png")
            result = analyze_file(str(p), debug_out=debug)
            row = result_row(str(p), result)
            rows.append(row)
            gefunden += row["nutri_score_present"]
            status = row["score"] if row["nutri_score_present"] else "KEIN"
            print(f"{p.name}: {status} (Konfidenz {row['confidence']})")
        except (ValueError, OSError) as exc:
            errors += 1
            print(f"{p.name}: FEHLER ({exc})", file=sys.stderr)

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FELDER)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV gespeichert: {csv_out}")
    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_out, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"JSONL gespeichert: {jsonl_out}")

    print(f"{gefunden}/{len(rows)} Bilder mit Nutri-Score, {errors} Fehler")
    return 0 if errors == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nutri-Score auf Verpackungsfronten erkennen")
    parser.add_argument("image", help="Pfad zu einem Bild oder einem Ordner mit Bildern")
    parser.add_argument("--debug", metavar="PNG", help="Annotiertes Bild speichern (nur Einzelbild)")
    parser.add_argument("--json", action="store_true", help="Rohergebnis als JSON ausgeben (nur Einzelbild)")
    parser.add_argument("--csv", metavar="CSV", type=Path, help="Batch: Ergebnistabelle als CSV")
    parser.add_argument("--jsonl", metavar="JSONL", type=Path, help="Batch: Ergebnisse als JSONL")
    parser.add_argument("--annotiert", metavar="ORDNER", type=Path,
                        help="Batch: Ordner fuer annotierte Kopien")
    args = parser.parse_args(argv)

    src = Path(args.image)
    if src.is_dir() or args.csv or args.jsonl or args.annotiert:
        return run_batch(src, args.csv, args.annotiert, args.jsonl)

    try:
        result = analyze_file(str(src), debug_out=args.debug)
    except (ValueError, OSError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        from .detector import format_result
        print(format_result(result))
    return 0 if result.get("nutri_score_present") else 1


if __name__ == "__main__":
    raise SystemExit(main())
