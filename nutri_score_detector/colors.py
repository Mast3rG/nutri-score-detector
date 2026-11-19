"""Offizielle Nutri-Score-Farben (RGB) und zugehörige HSV-Bereiche.

HSV-Werte gelten für OpenCV (H: 0-179, S/V: 0-255) und sind bewusst
leicht toleriert, um Schattierungen und JPEG-Artefakte abzudecken.
"""

COLORS: dict[str, dict] = {
    "A": {
        "name": "dunkelgruen",
        "hex": "#038141",
        "rgb": (3, 129, 65),
        "hsv_ranges": [((68, 130, 30), (90, 255, 235))],
    },
    "B": {
        "name": "hellgruen",
        "hex": "#85BB2F",
        "rgb": (133, 187, 47),
        "hsv_ranges": [((32, 100, 100), (51, 255, 255))],
    },
    "C": {
        "name": "gelb",
        "hex": "#FECB02",
        "rgb": (254, 203, 2),
        "hsv_ranges": [((19, 130, 140), (31, 255, 255))],
    },
    "D": {
        "name": "orange",
        "hex": "#EE8100",
        "rgb": (238, 129, 0),
        "hsv_ranges": [((11, 150, 120), (18, 255, 255))],
    },
    "E": {
        "name": "rot",
        "hex": "#E63E11",
        "rgb": (230, 62, 17),
        "hsv_ranges": [
            ((0, 150, 120), (10, 255, 255)),
            ((176, 150, 120), (180, 255, 255)),
        ],
    },
}

SCORE_LETTERS = ["A", "B", "C", "D", "E"]
