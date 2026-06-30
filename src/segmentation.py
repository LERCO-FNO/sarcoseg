import logging
import subprocess
from pathlib import Path
from time import perf_counter

import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from src import utils, visualization
from src.classes import (
    Centroids,
    ImageData,
    ProcessDurations,
    ProcessResult,
    SeriesSegmentationResult,
    StudyData,
    StudySegmentationResult,
)
from src.labels import DEFAULT_VERTEBRA_CLASSES

log = logging.getLogger("segment")

tissue_predictor = nnUNetPredictor()
tissue_predictor.initialize_from_trained_model_folder(
    "models/mazurowski_muscle_fat", use_folds=(5,)
)


def segment_ct_study(
    input_dir: str | Path,
    output_dir: str | Path,
    study_case: StudyData | None = None,
) -> StudySegmentationResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not study_case:
        study_case = StudyData._from_json(
            input_dir.joinpath(f"dicom_tags_{input_dir.name}.json")
        )

    series_nifti_filepaths = list(input_dir.rglob("input_ct_volume.nii.gz"))

    log.debug(
        f"found {len(series_nifti_filepaths)} volumes to segment spine in directory `{input_dir}`"
    )

    study_segmentation = StudySegmentationResult._from_study_case(study_case)

    for series_filepath in series_nifti_filepaths:
        series_output_dir = series_filepath.parent
        series_uid = series_output_dir.parts[-1]
        contrast_phase = study_case.series[series_uid].contrast_phase

        log.info(f"running segmentation series {series_uid}")

        spine_mask_data, spine_duration = segment_spine(
            series_filepath, series_output_dir
        )

        input_volume_data: ImageData = utils.read_volume(series_filepath)
        slice_extraction_result = extract_slices(
            input_volume_data.image,
            spine_mask_data.image,
            series_output_dir,
        )

        series_seg_result = SeriesSegmentationResult(
            series_uid=series_uid, contrast_phase=contrast_phase
        )

        if not slice_extraction_result:
            log.warning(
                f"participant {study_segmentation.participant}, study {study_segmentation.study_uid}, series {series_uid} has no L3 mask"
            )
            series_seg_result.status = ProcessResult.MISSING_L3_MASK
            study_segmentation.add_result(series_seg_result)
            continue

        tissue_volume_data, centroids, extraction_duration = slice_extraction_result

        tissue_seg_result = segment_tissues(tissue_volume_data.path, series_output_dir)

        if not tissue_seg_result:
            series_seg_result.status = ProcessResult.SEGMENTATION_TISSUE_FAIL
            study_segmentation.add_result(series_seg_result)
            continue

        tissue_mask_data, tissue_duration = tissue_seg_result

        processed_data, postproc_duration = utils.postprocess_tissue_masks(
            tissue_mask_data,
            tissue_volume_data,
        )

        metrics = utils.compute_metrics(
            processed_data,
            tissue_volume_data,
            patient_height=study_case.patient_height,
        )

        metrics.set_durations(
            ProcessDurations(
                spine_duration, tissue_duration, extraction_duration, postproc_duration
            )
        )

        metrics.centroids = centroids
        metrics.l3_slice_index = centroids.body_centroid[-1]
        metrics.image_z_size = input_volume_data.image.GetDepth()

        metrics.set_l3_tube_current(output_dir, series_uid)

        series_seg_result.metrics = metrics
        series_seg_result.status = ProcessResult.SEGMENTATION_FINISHED
        study_segmentation.add_result(series_seg_result)
        log.debug(f"segmentation finished for series {series_uid}")

        # case_images_dir = series_output_dir.joinpath("images")
        # case_images_dir.mkdir(exist_ok=True)
        visualization.overlay_spine_mask(
            input_volume_data.image,
            spine_mask_data.image,
            centroids,
            output_dir=series_output_dir,
        )

        visualization.overlay_tissue_mask(
            tissue_volume_data.image,
            processed_data.image,
            output_dir=series_output_dir,
        )
        log.debug("saved mask overlays")

    study_segmentation._write_to_json(output_dir)
    log.debug(f" {series_uid}")
    return study_segmentation


def segment_spine(
    input_nifti_path: str | Path,
    output_dir: str | Path = "./",
    vert_classes: list[str] | None = None,
    overwrite_output: bool = False,
) -> tuple[ImageData, float]:
    """
    Segment spine vertebrae. Skips segmentation if a spine_mask.nii.gz file exists.

    Args:
        input_nifti_path (Union[str, Path]): path to nifti file.
        output_dir (Union[str, Path], optional): directory to store segmentation mask.
            Defaults to `"./"`.
        vert_classes (list[str]): vertebrae class names in TotalSegmentator's task `total`.
            For class details see https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#class-details.

    Returns:
        tuple (ImageData, float): Tuple of segmented spine mask and segmentation duration.
        - **spine_mask** (ImageData): Spine segmentation mask.
        - **duration** (float): Duration of segmentation.
    """

    input_nifti_path = Path(input_nifti_path)
    output_dir = Path(output_dir)

    spine_mask_path = output_dir.joinpath("spine_mask.nii.gz")

    log.debug(f"segmenting vertebrae for {input_nifti_path.name}")

    if not vert_classes:
        vert_classes = list(DEFAULT_VERTEBRA_CLASSES.keys())
        log.debug(f"vert_classes is None, using default: {vert_classes}")

    start = perf_counter()

    command = [
        "TotalSegmentator",
        "-i",
        str(input_nifti_path),
        "-o",
        str(spine_mask_path),
        "--task",
        "total",
        "-ml",
        "-ot",
        "nifti",
        "--fast",
        "--device",
        "gpu",
        "--quiet",
        "--roi_subset",
    ] + list(vert_classes)

    res = subprocess.run(args=command, capture_output=True, text=True)
    duration = round(perf_counter() - start, ndigits=4)
    log.info(f"spine segmentation finised in {duration:.4f} seconds")

    return utils.read_volume(spine_mask_path), duration


def segment_tissues(
    tissue_volume_path: Path | str, case_output_dir: Path | str
) -> tuple[ImageData, float] | None:
    tissue_volume_path = Path(tissue_volume_path)
    case_output_dir = Path(case_output_dir)

    log.debug(f"starting tissue segmentation for {tissue_volume_path.name}")

    output_filepath = Path(case_output_dir, "tissue_mask.nii.gz")
    start = perf_counter()
    try:
        tissue_predictor.predict_from_files(
            list_of_lists_or_source_folder=[[str(tissue_volume_path)]],
            output_folder_or_list_of_truncated_output_files=[str(output_filepath)],
            num_processes_preprocessing=8,
            num_processes_segmentation_export=8,
        )
    except RuntimeError as err:
        log.error(f"tissue segmentation failed: {err}")
        return None

    duration = round(perf_counter() - start, ndigits=4)
    log.info(f"tissue segmentation finished in {duration:.4f} seconds")

    return utils.read_volume(output_filepath), duration


def extract_slices(
    ct_volume: sitk.Image,
    spine_mask: sitk.Image,
    output_dir: Path | str,
    slices_num: int = 0,
) -> tuple[ImageData, Centroids, float] | None:
    """
    Extract axial slices from input CT volume at vertebrae body's centroid.

    Args:
        ct_volume (Union[Nifti1Image, Path, str]): Input CT volume.
        spine_mask (Union[Nifti1Image, Path, str]): Segmented spine mask with labeled vertebrae.
        output_dir (Union[Path, str]): directory to save extracted slices data.
        slices_num (int, optional): Number of slices to extract. Defaults to 0. If slicing range extends CT volume size, it will be clipped to volume's size.

    Raises:
        FileNotFoundError: If input CT volume is not found.
        FileNotFoundError: If segmented spine mask is not found.

    Returns:
        tuple (ImageData, float): Tuple of segmented spine mask and segmentation duration.
        - **tissue_slice** (ImageData): Spine segmentation mask.
        - **centroids** (Centroids): Indexes of whole L3 vertebrae centroid and L3 vertebrae body centroid.
        - **duration** (float): Duration of segmentation.
    """
    if not isinstance(spine_mask, sitk.Image):
        raise TypeError(
            f"spine_mask should be of type sitk.Image, not `{type(spine_mask)}`"
        )

    if not isinstance(ct_volume, sitk.Image):
        raise TypeError(
            f"ct_volume should be of type sitk.Image, not `{type(ct_volume)}`"
        )

    start = perf_counter()
    centroids = utils.get_vertebrae_body_centroids(
        spine_mask, DEFAULT_VERTEBRA_CLASSES["vertebrae_L3"]
    )

    # only need to check for the whole vertebrae centroid
    if not centroids.vertebre_centroid:
        return None

    tissue_slice = ct_volume[
        ..., centroids.body_centroid[1] : centroids.body_centroid[1] + 1
    ]

    output_filepath = Path(output_dir, "tissue_slices.nii.gz")
    if output_filepath.exists():
        log.debug(f"file `{output_filepath}` exists, overwriting")

    # nib.save(tissue_slice, output_filepath)
    sitk.WriteImage(tissue_slice, output_filepath)

    duration = round(perf_counter() - start, ndigits=4)
    log.info(f"slice extraction finished in {duration:.4f} seconds")

    return (
        ImageData(image=tissue_slice, path=output_filepath),
        centroids,
        duration,
    )
