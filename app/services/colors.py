"""Skin tone + undertone → clothing colours that suit the person.

A simple, explainable version of the "colour season" idea used by stylists:

- UNDERTONE picks the colour family:
    warm    → earthy, golden-based colours (olive, mustard, rust, camel, cream, coral, teal)
    cool    → blue-based and jewel colours (navy, emerald, royal blue, burgundy, lavender, grey)
    neutral → a mix of softer colours from both (dusty rose, jade, taupe, denim, navy, sage)

- TONE DEPTH picks how light/strong the colours are, for good contrast with the skin:
    light skin (fair, light)   → softer and mid-tone colours; very pale colours can wash out
    medium skin (medium, tan)  → rich mid-tone colours
    deep skin (brown, deep)    → bright, saturated colours and crisp white, which pop

Each group has 8 colours with names a shopper recognises; several deliberately match the
names of colours in the store's catalogue. The store compares colours by LAB distance
(not by name), so names are for display only.
"""

from app.schemas import ColorSuggestion

DEPTH_GROUP = {
    "fair": "light",
    "light": "light",
    "medium": "medium",
    "tan": "medium",
    "brown": "deep",
    "deep": "deep",
}

_PALETTES: dict[tuple[str, str], list[tuple[str, str]]] = {
    # ---- warm ----
    ("warm", "light"): [
        ("Cream", "#F2E8D5"), ("Camel", "#C19A6B"), ("Coral", "#FF7F50"), ("Sage", "#9CAF88"),
        ("Sand", "#D8C8A8"), ("Teal", "#008080"), ("Peach", "#FFCBA4"), ("Olive", "#708238"),
    ],
    ("warm", "medium"): [
        ("Olive", "#708238"), ("Mustard", "#D4A017"), ("Rust", "#B7410E"), ("Camel", "#C19A6B"),
        ("Teal", "#008080"), ("Cream", "#F2E8D5"), ("Forest Green", "#228B22"), ("Chocolate", "#5C3A21"),
    ],
    ("warm", "deep"): [
        ("Mustard", "#D4A017"), ("Burnt Orange", "#CC5500"), ("Cream", "#F2E8D5"), ("Forest Green", "#228B22"),
        ("Gold", "#D4AF37"), ("Coral", "#FF7F50"), ("Teal", "#008080"), ("Camel", "#C19A6B"),
    ],
    # ---- cool ----
    ("cool", "light"): [
        ("Navy", "#1F2A44"), ("Sky Blue", "#87CEEB"), ("Lavender", "#B57EDC"), ("Rose Pink", "#E8A0BF"),
        ("Charcoal", "#36454F"), ("Heather Grey", "#9E9E9E"), ("White", "#F5F5F0"), ("Burgundy", "#800020"),
    ],
    ("cool", "medium"): [
        ("Navy", "#1F2A44"), ("Emerald", "#046307"), ("Royal Blue", "#4169E1"), ("Burgundy", "#800020"),
        ("Charcoal", "#36454F"), ("White", "#F5F5F0"), ("Plum", "#8E4585"), ("Maroon", "#800000"),
    ],
    ("cool", "deep"): [
        ("White", "#F5F5F0"), ("Royal Blue", "#4169E1"), ("Emerald", "#046307"), ("Fuchsia", "#C154C1"),
        ("Black", "#1A1A1A"), ("Cobalt", "#0047AB"), ("Ruby", "#9B111E"), ("Lavender", "#B57EDC"),
    ],
    # ---- neutral ----
    ("neutral", "light"): [
        ("Navy", "#1F2A44"), ("Dusty Rose", "#DCAE96"), ("Sage", "#9CAF88"), ("White", "#F5F5F0"),
        ("Heather Grey", "#9E9E9E"), ("Light Wash", "#7B9CC4"), ("Taupe", "#8B7D6B"), ("Teal", "#008080"),
    ],
    ("neutral", "medium"): [
        ("Navy", "#1F2A44"), ("Teal", "#008080"), ("Jade", "#00A86B"), ("Taupe", "#8B7D6B"),
        ("White", "#F5F5F0"), ("Indigo", "#2E4A7A"), ("Plum", "#8E4585"), ("Olive", "#708238"),
    ],
    ("neutral", "deep"): [
        ("White", "#F5F5F0"), ("Black", "#1A1A1A"), ("Jade", "#00A86B"), ("Cobalt", "#0047AB"),
        ("Coral", "#FF7F50"), ("Teal", "#008080"), ("Maroon", "#800000"), ("Mustard", "#D4A017"),
    ],
}  # fmt: skip


def suggest_colors(tone: str, undertone: str) -> list[ColorSuggestion]:
    depth = DEPTH_GROUP.get(tone, "medium")
    palette = _PALETTES.get((undertone, depth), _PALETTES[("neutral", depth)])
    return [ColorSuggestion(name=name, hex=hex_code) for name, hex_code in palette]
