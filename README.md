# Nutri-Score-Detektor

Analysiert Bilder von Verpackungsvorderseiten und beantwortet drei Fragen:

1. **Ist ein Nutri-Score vorhanden?**
2. **Welcher Score (A–E) ist es?**
3. **Wo auf der Vorderseite** (Bounding Box, Pixel + relative Koordinaten) **und in welcher Größe** (px, Anteil an Bildbreite/-höhe)?

## Funktionsweise

- **Farbsegmentierung**: Die fünf offiziellen Nutri-Score-Farben (`#038141`, `#85BB2F`, `#FECB02`, `#EE8100`, `#E63E11`) werden im HSV-Raum segmentiert (`nutri_score_detector/colors.py`). Kleine Bilder (< 400 px) werden zunächst hochskaliert.
- **Anker-basierte Kandidatensuche**: Um jede Farbkomponente werden die nächsten Nachbarn der anderen Farben gesucht, auch mit einer ausgelassenen Farbe (dann 4 Baender) und in 4 Orientierungen (0/90/180/270 Grad, für gedreht gestapelte Logos).
- **Verifikation über die offizielle Logo-Geometrie** (aus der Charta-SVG gemessen): 5 Farbbaender gleicher Höhe in einer Reihe; das **Score-Band ist eine Kapsel** – ca. 1,2-fache Breite und 1,25-fache Höhe der übrigen Baender und ragt oben/unten über die Bandreihe hinaus. Alle Nicht-Kapsel-Baender sind gleich hoch.
- **Score-Bestimmung**: Primärsignal ist die **Kapsel** (Breite/Höhe-Verhältnisse je Farbe). Rückfallebene: Größe der weißen Buchstaben-Aussparungen (`letter_margin` ≥ 2.0). Kandidaten ohne Kapsel- oder Buchstaben-Hinweis werden abgelehnt (Schutz vor zufällig bunten Grafiken).
- **Reinheit**: Stark überlappende Komponentenpaare (Segmentierungsartefakte beim Zusammenschmelzen benachbarter Baender) geben Abzug in der Rangfolge; Kandidaten werden parallel an der Original- und an der bereinigten Komponentenmenge bewertet.
- **Ausgabe**: Bounding Box in Pixel, relative Position/Größe, Konfidenz, Farb-Anteile (`bbox`, `bbox_relative`, `size` in der JSON-Ausgabe).

## Nutzung

```bash
pip install -r requirements.txt

python -m nutri_score_detector verpackung.jpg
python -m nutri_score_detector verpackung.jpg --json          # maschinenlesbare Ausgabe
python -m nutri_score_detector verpackung.jpg --debug out.png # annotiertes Bild
```

## Test/ Evaluation

```bash
# synthetische Verpackungen erzeugen (offizielle Logo-Geometrie):
python -m nutri_score_detector.make_test_images 60
python -m nutri_score_detector.evaluate
```

Aktuelle Ergebnisse auf 150 synthetischen Fronten inkl. Störfarben (seed 42, v6):

| Metrik | Wert |
|---|---|
| Presence-Genauigkeit | 100 % |
| Logo gefunden (bei vorhandenem Score) | 100 % |
| Score-Genauigkeit (wenn Logo gefunden) | 99,2 % |
| Mittlerer IoU der Bounding Box | 0,97 |

Über weitere Seeds (7, 123): Score-Genauigkeit 94–99 %, Presence ≥ 97,5 % —
bekannte Restfehler sind vor allem geschrumpfte/abgeschnittene Kapseln bei sehr
kleinen Logos.

### Echte Fotos (7 Packshots im `testdata/echtfoto_*`)

| Foto | Ergebnis | Score | Position | Bemerkung |
|---|---|---|---|---|
| echtfoto_01 | erkannt | D | vertikales Logo, links oben (Kapsel D) | per Zoom verifiziert: großes D |
| echtfoto_02 | erkannt | E | unten links | per Zoom verifiziert: großes E |
| echtfoto_03 | erkannt | A | unten links | helles B-Band nicht segmentierbar; Score unsicher, Position korrekt |
| echtfoto_04 | **korrekt nicht erkannt** | – | – | ÄTTA-Firmenlogo (bunte Buchstaben) -> Negativtest bestanden |
| echtfoto_05 | erkannt | A | links Mitte (sehr klein) | per Zoom verifiziert |
| echtfoto_06 | erkannt | C | unten links | Kapsel C |
| echtfoto_07 | erkannt | D | oben rechts Kapsel D | VLM-Ground-Truth unsicher (B oder D) |

## Wechsel auf echtes Lernen (empfohlen für reale Fotos)

Für echte Packshots (Beleuchtung, Schrägansicht, Crinkle) ist ein trainierter Objektdetektor robuster als Farbheuristik:

1. DATEN: Packshots sammeln (z. B. Open Food Facts – dort gibt es `nutriscore`-Tags und Frontenfotos), Logos per Bounding Box annotieren (CVAT/Label Studio).
2. TRAINING: YOLOv8/11 mit Klassen `nutriscore_a` … `nutriscore_e` trainieren (`ultralytics`).
3. ERSETZEN: `NutriScoreDetector.analyze()` durch den Modellaufruf ersetzen; die Ergebnis-Struktur (score, bbox, size) bleibt identisch – Auswertung und CLI funktionieren unverändert.
