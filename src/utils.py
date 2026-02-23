import json
import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import skimage as sk
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from pydicom import dcmread

from src import slogger
from src.classes import (
    ImageData,
    MetricsData,
    SeriesData,
    StudyData,
)

DEFAULT_VERTEBRA_CLASSES: dict[str, int] = {
    "vertebrae_L1": 31,
    "vertebrae_L2": 30,
    "vertebrae_L3": 29,
    "vertebrae_L4": 28,
    "vertebrae_L5": 27,
    "vertebrae_S1": 26,
}

DEFAULT_TISSUE_CLASSES: dict[str, int] = {
    "sat": 1,
    "vat": 2,
    "imat": 3,
    "muscle": 4,
}

TISSUE_LABEL_INDEX = list(DEFAULT_TISSUE_CLASSES.keys())

logger = slogger.get_logger(__name__)


def get_vertebrae_body_centroids(
    mask: Nifti1Image, vert_labels: int
) -> tuple[NDArray, NDArray]:
    """
    Get vertebrae's body centroid coordinates in pixel space.

    Args:
        mask (nib.nifti1.Nifti1Image): spine prediction mask
        vert_labels (int): vertebrae mask labels

    Returns:
        vert_body_centroid (NDArray):
            Vertebrae body centroid in voxel space.

        vert_centroid (NDArray):
            Vertebrae centroid in voxel space.
    """

    mask_arr: NDArray = mask.get_fdata().astype(np.uint8)

    # get sagittal slice at L3's center
    vert_label_centroid: NDArray = np.rint(
        sk.measure.centroid(mask_arr == vert_labels)
    ).astype(np.uint16)
    sagittal_slice_arr: NDArray = mask_arr[vert_label_centroid[0], ...]
    sagittal_l3 = np.where(sagittal_slice_arr == vert_labels, vert_labels, 0)

    # relabel L3 parts
    vert_components: NDArray = sk.measure.label(sagittal_l3)

    # get 2 largest components
    compoments_pixel_num: dict[int, int] = {
        prop.label: prop.num_pixels for prop in sk.measure.regionprops(vert_components)
    }
    largest_components_labels = sorted(
        compoments_pixel_num, key=compoments_pixel_num.get, reverse=True
    )

    # get centers of largest components
    comp_centroids = np.array(
        [
            sk.measure.centroid(vert_components == label)
            for label in largest_components_labels
        ]
    )
    comp_centroids = np.rint(comp_centroids).astype(np.uint16)

    """
    get the centroid in front of whole vertebrae mask centroid

    - axis directions/coordinates:
    vertebrae mask centroid: (X, Y, Z) -> (R, A, S)
    centroids: (Y, Z) -> (A, S), centroids numpy shapes (n, 2), where n is number of centroids

    - compare centroid coordinates in anterior (A) direction which increases towards anterior
    - comparison is done in pixel/voxel space
    """
    vert_body_centroid = comp_centroids[
        np.argmax(comp_centroids[:, 0] > vert_label_centroid[1])
    ]

    return vert_body_centroid, vert_label_centroid


def postprocess_tissue_masks(
    mask_data: ImageData,
    volume_data: ImageData,
):
    """
    store tissue labels in separate z-axis
    tissue      | label     | z-axis index
    muscle      | 1         | 0
    sat         | 2         | 1
    vat         | 3         | 2
    imat        | 4         | 3
    """
    start = perf_counter()
    mask_arr = mask_data.image.get_fdata().astype(np.uint8)
    volume_arr = volume_data.image.get_fdata()

    processed_mask = np.zeros(
        (*mask_arr.shape, len(DEFAULT_TISSUE_CLASSES)), dtype=bool
    )
    for i, label in enumerate(DEFAULT_TISSUE_CLASSES.values()):
        # for SAT (label == 1) use 200, for other tissues use 20
        min_hole_size = 200 if label == 1 else 20

        processed_mask[..., i] = sk.morphology.remove_small_holes(
            mask_arr == label, min_hole_size
        )

        if label == 4:
            # get muscle tissue pixels in HU
            muscle_hu = volume_arr * processed_mask[..., 3]

            imat_hu = np.logical_and(muscle_hu <= -30, muscle_hu >= -190)

            imat_hu_filt = sk.morphology.remove_small_objects(imat_hu, 10)
            processed_mask[imat_hu_filt, 2] = 1
            processed_mask[imat_hu_filt, 3] = 0

    # squeeze processed labels of shape H x W x D x L -> H x W x D
    out_nifti = np.zeros(processed_mask.shape[:-1], dtype=np.uint8)

    for i, label in enumerate(DEFAULT_TISSUE_CLASSES.values()):
        out_nifti[processed_mask[..., i]] = label
    out_nifti = nib.as_closest_canonical(
        nib.Nifti1Image(out_nifti, mask_data.image.affine)
    )
    processed_mask_path = mask_data.path.parent.joinpath("tissue_mask_pp.nii.gz")
    nib.save(out_nifti, processed_mask_path)  # overwrite segmented image

    duration = perf_counter() - start
    logger.info(f"tissue postprocessing finished in {duration:.2f} second")
    return ImageData(
        image=out_nifti, spacing=mask_data.spacing, path=processed_mask_path
    ), duration


def compute_metrics(
    tissue_mask_data: ImageData,
    tissue_volume_data: ImageData,
    patient_height: float | None = None,
) -> MetricsData:
    """Compute area and mean Hounsfield Unit for segmented tissue masks.
    Also compute skeletal muscle index (SMI) if `patient_height` is given.

    Units of computed metrics:
        - area: cm^2
        - mean_hu: HU
        - SMI: cm^2 / m^2

    Args:
        tissue_mask_data (ImageData): Segmented tissue masks of SAT, VAT, IMAT and MUSCLE.
        tissue_volume_data (ImageData): Input nifti volume.
        patient_height (float | None, optional): Patient's height. Defaults to None.

    Returns:
        metrics (MetricsData): Computed metrics.
    """
    if tissue_mask_data.spacing:
        spacing = tissue_mask_data.spacing
    elif tissue_volume_data.spacing:
        spacing = tissue_volume_data.spacing

    # if None use 1mm spacing in all directions
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    mask_arr = tissue_mask_data.image.get_fdata()

    depth = mask_arr.shape[-1]  # get image size only
    if depth == 1:
        pixel_size = np.prod(spacing[:2]) / 100.0  # pixel size is in cm^2
    elif depth > 1:
        pixel_size = np.prod(spacing) / 10000.0  # pixel size is in cm^3

    tissue_arr = tissue_volume_data.image.get_fdata()

    area = {
        tissue: np.count_nonzero(mask_arr == label) * pixel_size
        for tissue, label in DEFAULT_TISSUE_CLASSES.items()
    }
    mean_hu = {
        tissue: np.mean(tissue_arr[mask_arr == label])
        for tissue, label in DEFAULT_TISSUE_CLASSES.items()
    }

    smi = None
    if patient_height:
        # skeletal muscle index (smi) (cm^2 / m^2) = muscle area (cm^2) / patient height (m^2)
        # patient height is in cm^2
        smi = area["muscle"] / ((patient_height / 100.0) ** 2)
    return MetricsData(area=area, mean_hu=mean_hu, skelet_muscle_index=smi)


# TODO: maybe add verify_dicom_study()?? possibly no
def verify_dicom_study(study_inst_uid: str, dicom_filepath: str | Path):
    ds = dcmread(
        dicom_filepath,
        stop_before_pixels=True,
        specific_tags=["PatientID", "StudyInstanceUID"],
    )
    logger.info(
        f"verifying study instance uids:\nLabkey={study_inst_uid}\nDICOM={ds.StudyInstanceUID}"
    )
    return study_inst_uid == ds.StudyInstanceUID


def read_patient_list(
    filepath: str | Path, columns: list[str] | None = None
) -> pd.DataFrame:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.is_file() or not filepath.exists():
        logger.error(f"patient list at `{filepath}` is not a file or doesn't exist")
        raise FileNotFoundError(f"Patient list file not found at {filepath}")

    suffix = filepath.suffix
    if suffix == ".csv":
        df = pd.read_csv(
            filepath, index_col=False, header=0, dtype=str, usecols=columns
        )
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(
            filepath, index_col=False, header=0, dtype=str, usecols=columns
        )

    return df


def read_volume(path: Path | str):
    volume = nib.as_closest_canonical(nib.load(path))
    spacing = volume.header.get_zooms()
    return ImageData(image=volume, spacing=spacing, path=Path(path))


def remove_empty_segmentation_dir(dirpath: str | Path):
    logger.info(f"removing empty segmentation directory `{dirpath}`")
    shutil.rmtree(dirpath)


def remove_dicom_dir(dirpath: str | Path):
    logger.info(f"removing input DICOM directory `{dirpath}`")
    shutil.rmtree(dirpath)


def make_report(
    requested_study_cases: list[StudyData],
    output_dir: Path,
    timestamp: str | None = None,
    verbose: bool = False,
):
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    study_dirs = list(output_dir.glob("*"))
    missing_studies = [
        {"participant": pat.participant, "study_instance_uid": pat.study_inst_uid}
        for pat in requested_study_cases
        if output_dir.joinpath(pat.study_inst_uid) not in study_dirs
    ]

    finished_studies = [
        {
            "participant": pat.participant,
            "study_inst_uid": pat.study_inst_uid,
            "preprocessed_count": len(list(study_dir.rglob("input_ct_volume.nii.gz"))),
            "segmentated_count": len(list(study_dir.rglob("tissue_mask.nii.gz"))),
        }
        for pat in requested_study_cases
        if (study_dir := output_dir.joinpath(pat.study_inst_uid)).exists()
    ]

    report = {
        "timestamp": timestamp,
        "output_directory": str(output_dir.resolve()),
        "finished_studies": finished_studies,
        "missing_studies": missing_studies,
    }

    report_path = output_dir.joinpath(f"report_{timestamp}.json")
    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    logger.info(f"Segmentation report written to `{report_path}`")
