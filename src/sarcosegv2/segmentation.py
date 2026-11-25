from contextlib import contextmanager
from functools import lru_cache
from os import getenv
from pathlib import Path
from time import perf_counter

import SimpleITK as sitk
from loguru import logger as log
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from totalsegmentator.python_api import totalsegmentator

from sarcosegv2._types import StrPath
from sarcosegv2.data_io import read_from_json
from sarcosegv2.data_models import (
    Centroid,
    Result,
    SegmentationTime,
    SeriesResult,
    StudyMetadata,
    StudySegmentationResult,
)
from sarcosegv2.labels import TISSUE_HU_RANGES, TISSUE_LABEL_BY_NAME, VERTEBRAE_LABELS

MAZUROWSKI_TISSUE_MODEL = "models/mazurowski_muscle_fat"


class SegmentationError(Exception):
    pass


@lru_cache(maxsize=1)
def get_tissue_predictor(
    model_path: StrPath = MAZUROWSKI_TISSUE_MODEL,
) -> nnUNetPredictor:
    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(str(model_path), use_folds=(5,))
    log.debug(f"Loaded nnUNet model from `{model_path}`")
    return predictor


@contextmanager
def timed_step(durations: SegmentationTime, step_name: str):
    t0 = perf_counter()
    try:
        yield
    finally:
        setattr(durations, step_name, perf_counter() - t0)


def segment_ct(
    input_dir: StrPath, output_dir: StrPath, study_case: StudyMetadata | None = None
) -> tuple[Result, list]:

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if study_case is None:
        dicom_metadata_file = input_dir.joinpath(f"dicom_tags_{input_dir.name}.json")
        study_case = StudyMetadata.from_mapping(read_from_json(dicom_metadata_file))

    nifti_files = list(input_dir.rglob("input_volume.nii.gz"))
    log.debug(
        f"Found {len(nifti_files)} volumes for spine segmentation in `{input_dir}`"
    )

    study_result = StudySegmentationResult.from_study_metadata(study_case)
    log.info(f"Segmenting study case {study_result.participant}")

    # load predictor here, otherwise first tissue segmentation would include model load time
    predictor = get_tissue_predictor()

    failed = []
    for nifti_file in nifti_files:
        series_output_dir = nifti_file.parent
        series_metadata = study_case.series_by_name(series_output_dir.name)

        if series_metadata is None:
            log.error(
                f"No matching series metadata for nifti file `{series_output_dir}` found, skipping"
            )
            failed.append((input_dir, series_output_dir, Result.SeriesMetadataNotFound))
            continue

        series_name = series_metadata.series_name
        log.info(f"Segmenting spine for {study_case.participant} - {series_name}")

        series_result = SeriesResult(series_name, series_metadata.series_uid)

        # ------ SPINE SEGMENTATION ------
        output_spine_mask = series_output_dir.joinpath("spine_mask.nii.gz")
        try:
            with timed_step(series_result.durations, "spine_s"):
                segment_spine(nifti_file, output_spine_mask)
        except SegmentationError:
            log.exception(
                f"Failed spine segmentation for {series_metadata.series_name}"
            )
            series_result.status = Result.SpineSegmentationFail
            failed.append((input_dir, series_output_dir, series_result.status))
            continue

        # ------ EXTRACT SLICES ------
        output_tissue_slice = series_output_dir.joinpath("tissue_slice.nii.gz")
        result = extract_slices(nifti_file, output_spine_mask, output_tissue_slice)
        if result in (Result.ImageWrite, Result.VertebraeLabelNotFound):
            failed.append((input_dir, series_output_dir, result))
            continue

        # ------ TISSUE SEGMENTATION ------
        # TODO: remove after verifying predictor preload works
        # predictor = get_tissue_predictor()
        output_tissue_mask = series_output_dir.joinpath("tissue_mask.nii.gz")
        try:
            with timed_step(series_result.durations, "tissue_s"):
                segment_tissues(output_tissue_slice, output_tissue_mask, predictor)
        except SegmentationError:
            log.exception(
                f"Failed tissue segmentation for {series_metadata.series_name}"
            )
            series_result.status = Result.TissueSegmentationFail
            failed.append((input_dir, series_output_dir, series_result.status))
            continue

        # ------ TISSUE POSTPROCESS ------
        postprocess_tissue_mask(
            output_tissue_slice,
            output_tissue_mask,
        )

        series_result.status = Result.Ok

        study_result.series[series_name] = series_result
        log.debug(f"Finished series segmentation `{series_name}`")

    log.info(f"Case {study_case.participant} (study_uid {study_case.study_uid}):")
    log.info(
        f"Total NifTI files: {len(nifti_files)}, finished: {len(nifti_files) - len(failed)}, failed: {len(failed)}"
    )

    return Result.Ok, failed


def segment_spine(
    input_nifti: StrPath,
    output_nifti: StrPath,
):
    debug = getenv("DEBUG")
    is_verbose = True if debug else False

    log.debug(f"Segmenting spine of `{input_nifti}`")
    # start = perf_counter()
    try:
        totalsegmentator(
            input_nifti,
            output_nifti,
            ml=True,
            task="vertebrae_pp_refined",
            quiet=not is_verbose,  # inverse the verbosity
            verbose=is_verbose,
            debug=is_verbose,
            device="gpu",
        )
    except RuntimeError as err:
        raise SegmentationError(f"Spine segmentation failed for {input_nifti}") from err
    # elapsed = perf_counter() - start
    # log.debug(f"Spine segmentation finished in {elapsed:.4f} seconds")


def segment_tissues(
    tissue_slice_path: StrPath,
    tissue_mask_path: StrPath | None = None,
    predictor: nnUNetPredictor | None = None,
) -> float | None:
    tissue_slice_path = Path(tissue_slice_path)
    if tissue_mask_path:
        tissue_mask_path = Path(tissue_mask_path)
    else:
        # FIXME: verify correct path pointing
        # figure out better of way of accepting StrPath | sitk.Image - or accept only StrPath
        # includes extract_slices(), segment_tissues(), segment_spine(), postprocess_tissue_mask()
        tissue_mask_path = tissue_slice_path.parent.joinpath("tissue_mask.nii.gz")

    if predictor is None:
        predictor = get_tissue_predictor()

    log.debug(f"Segmenting tissues of `{tissue_slice_path}`")
    start = perf_counter()
    try:
        predictor.predict_from_files(
            list_of_lists_or_source_folder=[[str(tissue_slice_path)]],
            output_folder_or_list_of_truncated_output_files=[str(tissue_mask_path)],
        )
    except RuntimeError as err:
        log.exception("Tissue segmentation failed:")
        log.exception(f"{err}")
        return None
    elapsed = perf_counter() - start
    log.debug(f"Tissue segmentation finished in {elapsed:.4f}")
    log.debug(f"Segmented tissue masks written to `{tissue_mask_path}`")
    return elapsed


def extract_slices(
    ct_volume: StrPath | sitk.Image,
    spine_mask: StrPath | sitk.Image,
    output_tissue_slice: StrPath | None = None,
) -> Centroid | Result:
    """
    Extract slice from CT volume at vertebrae body centroid z-index (axial plane).
    """

    if isinstance(ct_volume, (str, Path)):
        ct_volume = sitk.ReadImage(ct_volume)
    if isinstance(ct_volume, (str, Path)):
        spine_mask = sitk.ReadImage(spine_mask)
    if output_tissue_slice:
        output_tissue_slice = Path(output_tissue_slice)
    else:
        output_tissue_slice = Path.cwd().joinpath("tissue_slice.nii.gz")

    centroid = find_vertebrae_body_centroid(
        spine_mask, VERTEBRAE_LABELS["vertebrae_L3"]
    )

    if centroid is None:
        return Result.VertebraeLabelNotFound

    # save as 3D image of dimensions (x, y, 1)
    tissue_slice = ct_volume[..., centroid.z : centroid.z + 1]

    # output_tissue_slice = output_tissue_slice.joinpath("tissue_slice.nii.gz")
    if output_tissue_slice.exists():
        log.warning(f"Overwriting `{output_tissue_slice}`")
    try:
        sitk.WriteImage(tissue_slice, output_tissue_slice)
        log.info(
            f"Written tissue slice (at index {centroid}) into `{output_tissue_slice}`"
        )
    except RuntimeError as err:
        log.exception(f"Failed writing tissue slice at into `{output_tissue_slice}`")
        log.exception(f"{err}")
        return Result.ImageWrite

    return centroid


def find_vertebrae_body_centroid(
    spine_mask: sitk.Image, vertebrae_label: int
) -> Centroid | None:
    """
    Return centroid coordinates in voxel space of vertebrae body for given label value.
    """
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(spine_mask)

    if vertebrae_label not in stats_filter.GetLabels():
        log.error(f"Mask label vertebrae_L3 (label {vertebrae_label}) not found")
        return None

    phys_coords = stats_filter.GetCentroid(vertebrae_label)
    return Centroid(**spine_mask.TransformPhysicalPointToIndex(phys_coords))


def postprocess_tissue_mask(tissue_slice: StrPath, tissue_mask: StrPath):
    """
    Relabel IMAT and VAT masks within specific HU range:
    `muscle` mask labels with HU values within (-190, -30) (both inclusive) replaced with `imat` mask label.
    `vat` mask labels with HU values greater than -51 (exclusive) replaced with 0.

    Both relabeling preceeded with morphological BinaryOpeningByReconstruction step to remove small objects/blobs.
    Uses structuring element sitkBall of size 1.
    """

    tissue_mask = Path(tissue_mask)

    imat_hu_range = TISSUE_HU_RANGES["imat"]
    vat_hu_range = TISSUE_HU_RANGES["vat"]

    mask_image = sitk.ReadImage(tissue_mask)
    tissue_image = sitk.ReadImage(tissue_slice)

    # During testing identified mismatch between mask-tissue span in physical space.
    # Resample to make sure both images occupy same physical space.
    if mask_image.GetOrigin() != tissue_image.GetOrigin():
        log.warning(
            "Mask and tissue images do not occupy same physical space, resampling mask..."
        )

        mask_image = sitk.Resample(
            mask_image,
            referenceImage=tissue_image,
            interpolator=sitk.sitkNearestNeighbor,
            defaultPixelValue=0,
            outputPixelType=mask_image.GetPixelID(),
        )

    imat_thresh = (imat_hu_range[0] <= tissue_image <= imat_hu_range[1]) & (
        mask_image == TISSUE_LABEL_BY_NAME["muscle"]
    )
    imat_thresh = sitk.BinaryOpeningByReconstruction(imat_thresh)
    mask_image[imat_thresh] = TISSUE_LABEL_BY_NAME["imat"]

    non_vat_thresh = (tissue_image > vat_hu_range[1]) * (
        mask_image == TISSUE_LABEL_BY_NAME["vat"]
    )
    non_vat_thresh = sitk.BinaryOpeningByReconstruction(non_vat_thresh)
    mask_image[non_vat_thresh] = 0

    output_pp_mask = tissue_mask.parent.joinpath("tissue_mask_pp.nii.gz")
    sitk.WriteImage(mask_image, output_pp_mask)
    log.info(f"Written postprocessed mask to `{output_pp_mask}`")
