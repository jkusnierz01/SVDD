LABEL_MAP = {"bonafide": 0, "spoof": 1, "deepfake": 1}
LABEL_MAP_M6 = {"human": 0, "ai": 1}

def base_stem(stem: str) -> str:
    return stem.split("__seg", 1)[0]

def parse_split_label_singfake(stem: str):
    base = base_stem(stem)
    parts = base.split("_")
    split = parts[0]
    label = parts[-1].lower()
    return split, LABEL_MAP[label]


def parse_split_label(stem: str):
    parts = stem.split("_")
    split = parts[0]
    label = parts[-1].lower()
    return split, LABEL_MAP[label]


def parse_split_label_mom(stem: str):
    parts = stem.split("_")
    split = "test"
    label = parts[0]
    return split, LABEL_MAP[label]


def parse_split_label_m6(stem: str):
    parts = stem.split("_")
    split = "test"
    label = parts[0]
    return split, LABEL_MAP_M6[label]



