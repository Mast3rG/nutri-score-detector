"""Offizielle Nutri-Score-Farben (RGB) und zugehörige HSV-Bereiche.

HSV-Werte gelten für OpenCV (H: 0-179, S/V: 0-255) und sind bewusst
tolerant, um Beleuchtung, Entsaettigung und JPEG-Artefakte abzudecken.
"""

COLORS: dict[str, dict] = {
    "A": {
        "name": "dunkelgruen",
        "hex": "#038141",
        "rgb": (3, 129, 65),
        "hsv_ranges": [((62, 60, 40), (96, 255, 240))],
    },
    "B": {
        "name": "hellgruen",
        "hex": "#85BB2F",
        "rgb": (133, 187, 47),
        "hsv_ranges": [((32, 60, 90), (54, 255, 255))],
    },
    "C": {
        "name": "gelb",
        "hex": "#FECB02",
        "rgb": (254, 203, 2),
        "hsv_ranges": [((17, 60, 140), (33, 255, 255))],
    },
    "D": {
        "name": "orange",
        "hex": "#EE8100",
        "rgb": (238, 129, 0),
        "hsv_ranges": [((9, 80, 120), (17, 255, 255))],
    },
    "E": {
        "name": "rot",
        "hex": "#E63E11",
        "rgb": (230, 62, 17),
        "hsv_ranges": [
            ((0, 80, 120), (9, 255, 255)),
            ((168, 80, 120), (180, 255, 255)),
        ],
    },
}

SCORE_LETTERS = ["A", "B", "C", "D", "E"]
