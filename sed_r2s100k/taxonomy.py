"""R2S100K class IDs and prompt vocabulary."""
# Source: R2S100K Table I and Sec. III-C1; upstream utils/helpers.py
# defines bg as ID 0 / RGB (0, 0, 0). The paper does not provide a
# separate exhaustive background definition. The complement of its 14
# target classes and the visual examples below are our prompt interpretation.
BACKGROUND_DESCRIPTION = (
    "Scene outside the annotated road-region surface: sky, hillsides, trees, "
    "buildings, walls and other surroundings beyond the road. Objects that "
    "are not one of the fourteen road-region categories may also be background "
    "even when they appear on the roadway. Roadside vegetation annotated as "
    "Vegetation Misc remains class 7. ID 0 is evaluated, not ignored."
)
# SED's VOC-background evaluation predicts concrete scene categories and maps
# the extra classes to background after argmax. Keep these concepts separate
# instead of averaging unrelated objects into one text embedding.
BACKGROUND_PROMPT_VARIANTS = (
    "sky",
    "tree",
    "mountain hillside",
    "building",
    "wall beside a road",
    "vehicle",
    "person",
    "roadside barrier",
)

CLASSES = {
    0: ("Background", BACKGROUND_DESCRIPTION, (0, 0, 0)),
    1: (
        "Wet Surface",
        "Slightly watered road surface; it can be damp due to snow or cold weather.",
        (2, 79, 59),
    ),
    2: (
        "Asphalt",
        "Road pavement constructed using aggregates such as crushed rocks, sand, and coal tar.",
        (17, 163, 74),
    ),
    3: ("Boggy", "Unpaved road surface filled with mud.", (112, 84, 62)),
    4: (
        "Earthen",
        "Unpaved road with compacted layers of stabilized soil.",
        (225, 148, 79),
    ),
    5: (
        "Crag-stone",
        "Hilltop stones dropped over the road surface in mountainous areas.",
        (120, 114, 104),
    ),
    6: ("Road Grime", "Dirt ingrained on the road surface.", (166, 130, 95)),
    7: (
        "Vegetation Misc",
        "Naturally occurring vegetation other than trees adjacent to the road.",
        (128, 222, 91),
    ),
    8: (
        "Distress",
        "Longitudinal and transverse cracks on the road caused by lack of maintenance.",
        (119, 61, 128),
    ),
    9: (
        "Drainage Grate",
        "An elongated perforated cover or grating used to cover a water drain.",
        (93, 86, 176),
    ),
    10: ("Water Puddle", "A small pool of water over the road.", (140, 160, 222)),
    11: (
        "Speed Breaker",
        "Concrete speed bumps, speed humps, and speed cushions over the road surface.",
        (234, 133, 5),
    ),
    12: ("Misc", "An unclear object dropped over the road.", (156, 28, 39)),
    13: (
        "Gravel",
        "Unpaved surface with loose aggregates of variable-sized rock fragments.",
        (99, 122, 130),
    ),
    14: (
        "Concrete",
        "Road material made with binders such as rough and fine aggregates.",
        (123, 43, 31),
    ),
}

BACKGROUND_RGB = CLASSES[0][2]

NUM_CLASSES = 15

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SHORT_PROMPTS = {
    0: "scene background outside road-region categories",
    1: "wet road surface",
    2: "asphalt road",
    3: "muddy road",
    4: "dirt road",
    5: "rocks on the road",
    6: "dirt on the road",
    7: "roadside vegetation",
    8: "cracks in the road",
    9: "drainage grate",
    10: "puddle on the road",
    11: "speed bump",
    12: "object on the road",
    13: "gravel road",
    14: "concrete road",
}

DESCRIPTION_PROMPTS = {
    1: "slightly wet road surface",
    2: "asphalt road pavement",
    3: "muddy unpaved road",
    4: "compacted earthen road",
    5: "loose stones dropped on a mountain road",
    6: "dirt ingrained in the road surface",
    7: "roadside vegetation other than trees",
    8: "longitudinal and transverse cracks in road pavement",
    9: "elongated perforated drainage grate",
    10: "small puddle on the road",
    11: "concrete speed bump across the road",
    12: "unclear discarded object lying on the road",
    13: "unpaved gravel surface with loose rock fragments",
    14: "concrete road material with coarse and fine aggregate",
}


def model_vocabulary():
    """Return 14 road concepts followed by visual background concepts."""
    expected = set(range(NUM_CLASSES))
    if set(CLASSES) != expected or set(DESCRIPTION_PROMPTS) != expected - {0}:
        raise ValueError("Taxonomy must define IDs 0–14 and prompts for IDs 1–14")
    if len({CLASSES[i][2] for i in range(NUM_CLASSES)}) != NUM_CLASSES:
        raise ValueError("Taxonomy colors must be unique")
    prompts = [DESCRIPTION_PROMPTS[i] for i in range(1, NUM_CLASSES)] + list(BACKGROUND_PROMPT_VARIANTS)
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("Every class must have a nonempty SED prompt")
    return prompts


def model_to_dataset_ids():
    """Map SED output channels to the 15 R2S100K mask IDs."""
    return list(range(1, NUM_CLASSES)) + [0] * len(BACKGROUND_PROMPT_VARIANTS)
