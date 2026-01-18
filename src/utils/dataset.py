LABEL_MAP = {"bonafide": 0, "spoof": 1, "deepfake": 1}

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



