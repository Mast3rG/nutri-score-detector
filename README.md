# Nutri-Score-Detektor

Analysiert Bilder von Verpackungsvorderseiten und beantwortet drei Fragen:

1. **Ist ein Nutri-Score vorhanden?**
2. **Welcher Score (A–E) ist es?**
3. **Wo auf der Vorderseite** (Bounding Box, Pixel + relative Koordinaten) **und in welcher Größe** (px, Anteil an Bildbreite/-höhe)?

## Funktionsweise

- **Farbsegmentierung**: Die fünf offiziellen Nutri-Score-Farben (`#038141`, `#85BB2F`, `#FECB02`, `#EE8100`, `#E63E11`) werden im HSV-Raum segmentiert (`nutri_score_detector/colors.py`).
- **Kandidatenregionen**: Farbmasken werden morphologisch verbunden; Regionen, in denen mindestens 4 der 5 Farben nennenswert vertreten sind, gelten als Nutri-Score-Logo.
- **Score-Bestimmung**: Primärsignal ist der **vergrößerte weiße Buchstabe** des aktiven Abschnitts (`filled - band`-Fläche je Farbband). Tie-Breaker ist die Geometrie: der **letzte vollhöhe Farbbalken** entspricht dem Score (Abschnitte nach dem Score sind im Logo verkleinert).
- **Ausgabe**: Bounding Box in Pixel, relative Position/Größe, Konfidenz, Farb-Anteile.

## Nutzung

```bash
pip install -r requirements.txt

python -m nutri_score_detector verpackung.jpg
python -m nutri_score_detector verpackung.jpg --json          # maschinenlesbare Ausgabe
python -m nutri_score_detector verpackung.jpg --debug out.png # annotiertes Bild
```

## Test/ Evaluation

```bash
python -m nutri_score_detector.make_test_images 60   # 60 synthetische Verpackungen in testdata/
python -m nutri_score_detector.evaluate              # Presence/Score/IoU-Auswertung
```

Aktuelle Ergebnisse auf 150 synthetischen Fronten inkl. Störfarben:

| Metrik | Wert |
|---|---|
| Presence-Genauigkeit | 99,3 % |
| Logo gefunden (bei vorhandenem Score) | 99,1 % |
| Score-Genauigkeit (wenn Logo gefunden) | 100 % |
| Mittlerer IoU der Bounding Box | 0,91 |

Bei JPEG 65 + Gauß-Rauschen + Rotation ±6° noch ~97 % Presence / ~97 % Score.

## Wechsel auf echtes Lernen (empfohlen für reale Fotos)

Für echte Packshots (Beleuchtung, Schrägansicht, Crinkle) ist ein trainierter Objektdetektor robuster als Farbheuristik:

1. DATEN: Packshots sammeln (z. B. Open Food Facts – dort gibt es `nutriscore`-Tags und Frontenfotos), Logos perBounding Box annotieren (CVAT/Label Studio).
2. TRAINING: YOLOv8/11 mit Klassen `nutriscore_a` … `nutriscore_e` trainieren (`ultralytics`).
3. ERSETZEN: `NutriScoreDetector.analyze()` durch Aufruf des Modells behalten; die Ergebnis-Struktur (score, bbox, size) bleibt identisch – Auswertung und CLI funktionieren unverändert.
