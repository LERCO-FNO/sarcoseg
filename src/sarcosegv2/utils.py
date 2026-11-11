from pathlib import Path


def find_dicoms(dicom_dir: Path) -> tuple[list[Path], Path] | None:
    """
    Discover DICOM files excluding DICOMDIR file.

    Parameters
    ----------
    dicom_dir : Path
        Path to DICOM study directory.

    Returns
    -------
    paths : list(Path)
        List of discovered DICOM file paths, otherwise `None`.
    root : Path
        Root path of study directory.
    """

    for root, dirs, files in dicom_dir.walk():
        if len(files) == 0 or "DICOMDIR" in files:
            continue
        paths = [f for f in root.iterdir() if f.is_file()]

        if not paths:
            return None
        return paths, root
