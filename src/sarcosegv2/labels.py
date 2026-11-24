VERTEBRAE_LABELS: dict[str, int] = {"vertebrae_L3": 22}

TISSUE_HU_RANGES: dict[str, tuple[int, int]] = {
    "muscle": (-29, 150),
    "imat": (-190, -30),
    "vat": (-205, -51),
}

TISSUE_LABEL_BY_NAME: dict[str, int] = {
    "muscle": 1,
    "sat": 2,
    "vat": 3,
    "imat": 4,
}

TISSUE_NAME_BY_LABEL: dict[int, str] = {
    1: "muscle",
    2: "sat",
    3: "vat",
    4: "imat",
}
