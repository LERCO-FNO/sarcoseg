import subprocess
from pathlib import Path
from time import perf_counter

import nibabel as nib
from nibabel import Nifti1Image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from src import utils
from src.utils import DEFAULT_VERTEBRA_CLASSES
from src import visualization
from src.slogger import get_logger
from src.classes import ImageData, SegmentationResult, StudyData, Centroids


logger = get_logger(__name__)

MODEL_DIR = Path("models", "muscle_fat_tissue_stanford_0_0_2")

tissue_predictor = nnUNetPredictor()
tissue_predictor.initialize_from_trained_model_folder(
    "models/mazurowski_muscle_fat", use_folds=(5,)
)


def segment_ct_study(
    input_dir: str | Path,
    output_dir: str | Path,
    study_case: StudyData | None = None,
    slices_num: int = 0,
    save_mask_overlays: bool = False,
) -> SegmentationResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not study_case:
        study_case = StudyData._from_json(
            input_dir.joinpath(f"dicom_tags_{input_dir.name}.json")
        )

    series_nifti_filepaths = list(input_dir.rglob("input_ct_volume.nii.gz"))
    logger.info("-" * 25)
    logger.info(
        f"found {len(series_nifti_filepaths)} volumes to segment spine in directory `{input_dir}`"
    )

    seg_result = SegmentationResult._from_study_case(study_case)

    for series_filepath in series_nifti_filepaths:
        input_volume_data: ImageData = utils.read_volume(series_filepath)

        series_output_dir = series_filepath.parent
        series_inst_uid = series_output_dir.parts[-1]

        logger.info(
            f"running segmentation on CT series {series_inst_uid} of participant {seg_result.participant}"
        )

        spine_mask_data, spine_duration = segment_spine(
            series_filepath, series_output_dir
        )

        tissue_volume_data, centroids, extraction_duration = extract_slices(
            input_volume_data.image,
            spine_mask_data.image,
            series_output_dir,
            slices_num,
        )

        tissue_mask_data, tissue_duration = segment_tissues(
            tissue_volume_data.path, series_output_dir
        )

        # TODO check this and verify if it should be used
        processed_data, postproc_duration = utils.postprocess_tissue_masks(
            tissue_mask_data,
            tissue_volume_data,
        )

        # TODO maybe replace skimage with simpleitk?
        metrics = utils.compute_metrics(
            processed_data,
            tissue_volume_data,
            patient_height=study_case.patient_height,
        )

        metrics.set_duration(
            spine_duration, tissue_duration, extraction_duration, postproc_duration
        )
        metrics.centroids = centroids
        metrics.series_inst_uid = series_inst_uid
        metrics.contrast_phase = study_case.series[series_inst_uid].contrast_phase

        seg_result.metrics_dict[series_inst_uid] = metrics

        if save_mask_overlays:
            case_images_dir = series_output_dir.joinpath("images")
            case_images_dir.mkdir(exist_ok=True)
            visualization.overlay_spine_mask(
                input_volume_data.image,
                spine_mask_data.image,
                centroids.vertebre_centroid,
                output_dir=case_images_dir,
            )

            visualization.overlay_tissue_mask(
                tissue_volume_data.image,
                processed_data.image,
                output_dir=case_images_dir,
            )

    seg_result._write_to_json(output_dir)
    return seg_result


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

    logger.info(f"\nsegmenting vertebrae for {input_nifti_path.name}")

    if not vert_classes:
        vert_classes = list(DEFAULT_VERTEBRA_CLASSES.keys())
        logger.info(f"vert_classes is None, using default: {vert_classes}")

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

    subprocess.run(args=command)

    duration = perf_counter() - start
    logger.info(f"spine segmentation finised in {duration:.2f} seconds")

    spine_mask = nib.as_closest_canonical(nib.load(spine_mask_path))
    spacing = spine_mask.header.get_zooms()
    return ImageData(image=spine_mask, path=spine_mask_path, spacing=spacing), duration


def segment_tissues(
    tissue_volume_path: Path | str, case_output_dir: Path | str
) -> tuple[ImageData, float]:
    tissue_volume_path = Path(tissue_volume_path)
    case_output_dir = Path(case_output_dir)

    logger.info(f"\nstarting tissue segmentation for {tissue_volume_path.name}")

    output_filepath = Path(case_output_dir, "tissue_mask.nii.gz")
    print(f"{tissue_volume_path=}")
    start = perf_counter()
    try:
        tissue_predictor.predict_from_files(
            list_of_lists_or_source_folder=[[str(tissue_volume_path)]],
            output_folder_or_list_of_truncated_output_files=[str(output_filepath)],
            num_processes_preprocessing=8,
            num_processes_segmentation_export=8,
        )
    except RuntimeError:
        logger.info(f"nnUNet finished `{tissue_volume_path}`")

    duration = perf_counter() - start
    logger.info(f"tissue segmentation finished in {duration}")

    tissue_mask: ImageData = utils.read_volume(output_filepath)
    return tissue_mask, duration


def extract_slices(
    ct_volume: Nifti1Image | Path | str,
    spine_mask: Nifti1Image | Path | str,
    output_dir: Path | str,
    slices_num: int = 0,
) -> tuple[ImageData, Centroids, float]:
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

    """
    if not isinstance(spine_mask, Nifti1Image):
        if Path(spine_mask).exists():
            spine_mask = nib.as_closest_canonical(nib.load(spine_mask))
        else:
            raise FileNotFoundError(spine_mask)

    if not isinstance(ct_volume, Nifti1Image):
        if Path(ct_volume).exists():
            ct_volume = nib.as_closest_canonical(nib.load(ct_volume))
        else:
            raise FileNotFoundError(ct_volume)

    start = perf_counter()
    body_centroid, vert_centroid = utils.get_vertebrae_body_centroids(
        spine_mask, DEFAULT_VERTEBRA_CLASSES["vertebrae_L3"]
    )

    # requires slices_num=2 at minimum
    if slices_num >= 2:
        slices_range = [
            # extract slices in Z direction (superior-inferior)
            body_centroid[-1] - (slices_num // 2),
            body_centroid[-1] + (slices_num // 2),
        ]
    else:
        slices_range = [body_centroid[-1], body_centroid[-1]]

    slices_range[-1] += 1  # nibabel slicer requires range [..., i:i + 1]

    if slices_range[0] < 0:
        slices_range[0] = 0
        logger.info(
            f"lower index {slices_range[0]} is outside lower extent for Z dimension, setting to 0"
        )

    z_size = ct_volume.shape[-1]
    if slices_range[1] > z_size:
        slices_range[1] = z_size
        slices_range[0] -= 1
        logger.info(
            f"upper index {slices_range[1]} is outside upper extent for Z dimension {z_size}, setting to {z_size}"
        )

    logger.info(
        f"extracting {slices_range[1] - slices_range[0]} slices in range {slices_range}"
    )

    sliced_volume = ct_volume.slicer[..., slices_range[0] : slices_range[1]]
    sliced_volume = nib.as_closest_canonical(sliced_volume)

    output_filepath = Path(output_dir, "tissue_slices.nii.gz")
    if output_filepath.exists():
        logger.info(f"file `{output_filepath}` exists, overwriting")

    try:
        nib.save(sliced_volume, output_filepath)
    except RuntimeError as err:
        logger.error(err)

    duration = perf_counter() - start
    logger.info(f"slice extraction finished in {duration} seconds")

    return (
        ImageData(image=sliced_volume, path=output_filepath),
        Centroids(
            vertebre_centroid=vert_centroid.tolist(),
            body_centroid=body_centroid.tolist(),
        ),
        duration,
    )
