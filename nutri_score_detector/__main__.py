"""Kommandozeile: Nutri-Score eines Bildes pruefen.

Beispiel:
    python -m nutri_score_detector pfad/zu/verpackung.jpg --debug debug.png
"""

from __future__ import annotations

import argparse
import json
import sys

from .detector import analyze_file, format_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nutri-Score auf Verpackungsfront erkennen")
    parser.add_argument("image", help="Pfad zum Bild der Verpackungsvorderseite")
    parser.add_argument("--debug", metavar="PNG", help="Annotiertes Bild speichern")
    parser.add_argument("--json", action="store_true", help="Rohergebnis als JSON ausgeben")
    args = parser.parse_args(argv)

    try:
        result = analyze_file(args.image, debug_out=args.debug)
    except (ValueError, OSError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_result(result))
    return 0 if result.get("nutri_score_present") else 1


if __name__ == "__main__":
    raise SystemExit(main())
