import re
from collections import defaultdict
from collections.abc import Iterable

from loguru import logger as log
from pydicom import Dataset

INVALID_BODY_PART_PATTERN = ["spine", "head", "heart"]
INVALID_IMAGE_TYPE_PATTERN = ["secondary", "localizer", "scout"]
INVALID_SERIES_DESC_PATTERN = re.compile(
    r"|".join(
        (
            "protocol",
            "topogram",
            "scout",
            "patient",
            "dose",
            "report",
            "monitor",
            "text",
            # "planning",
            "mip",
            "line",
            "distance",
            "head",
            "coronal",
            "cor",
            "sag",
            "sagital",
            "sagittal",
            # "bestdiast",
            # "bestsyst",
            "thick",
            "result",
        )
    ),
    re.IGNORECASE,
)

NATIVE_DESC_PATTERN = [
    "abdomen",
    "colo",
    "hip",
    "late",
    "ldct",
    "lung",
    "osteo",
    "pelvis",
    "skelet",
    "spine",
    "spiral",
    "thorabd",
    "thorax",
    "telo-femory",
    "wholebody",
]

ARTERIAL_DESC_PATTERN = [
    "arterial",
    "aorta",
    "aortic",
    "angio",
    "cta",
    "chestpain",
    "tavi",
    "bestsyst",
    "bestdiast",
    "chestcasc",
]

VENOUS_DESC_PATTERN = [
    "venous",
]

NEPHRO_DESC_PATTERN = [
    "nephro",
]

CONTRAST_PHASE_DESC_PATTERNS = {
    "abdomen": NATIVE_DESC_PATTERN,
    "arterial": ARTERIAL_DESC_PATTERN,
    "venous": VENOUS_DESC_PATTERN,
    "nephro": NEPHRO_DESC_PATTERN,
}


def filter_cfind_series_responses(
    datasets: Iterable[Dataset],
) -> tuple[defaultdict[str, list[Dataset]], list[Dataset] | None]:
    """
    Filter datasets in sequential order by:
    1. SeriesDescription - reject non-matching keywords, otherwise pass
    2. ImageType - reject non-matching keywords, otherwise pass
    3. SliceThickness - reject those without the tag, otherwise pass
    3. ConvolutionKernel - reject non-matching keywords, otherwise pass
    4. BodyPartExamined

    Each rule will reject a dataset on non-matching keywords, otherwise passed to next rule.
    Valid dataset passes all rules.

    Parameters
    ----------
    datasets : Iterable(pydicom.Dataset)
        SERIES level response datasets.

    Returns
    -------
    datasets_by_uid : dict(str, list(pydicom.Dataset))
        Filtered response datasets mapped by SeriesInstanceUID.
    dose_files : list(Dataset) or None
        Dose files if found by "dose report" in SeriesDescription, otherwise None.
    """
    # image_files = []
    datasets_by_uid: defaultdict[str, list[Dataset]] = defaultdict(list[Dataset])
    dose_files = []
    for ds in datasets:
        desc = ds.get("SeriesDescription", "")

        msg = "Filter `{keyword}` {decision} series `{desc}` due to: {reason}"

        # 1. Get Dose Report series first, we don't care about anything else
        if "dose report" in desc.lower():
            dose_files.append(ds)
            log.debug(
                msg.format(
                    keyword="SeriesDescription",
                    desc=desc,
                    decision="accepted",
                    reason="Dose Report match",
                )
            )
            continue

        # 2. Reject any SECONDARY or LOCALIZER series - reconstructed images, scout/localizer/topogram x-ray.
        # if ImageType has any of ["ORIGINAL", "PRIMARY", "AXIAL"] series is assumed a valid full resolution volume,
        # otherwise secondary/derived/other image type or non-image type with ImageType tag
        if image_type := ds.get("ImageType", None):
            image_type = [i.lower() for i in image_type]
            matches = [p for p in INVALID_IMAGE_TYPE_PATTERN if p in image_type]
            if matches:
                log.debug(
                    msg.format(
                        keyword="ImageType",
                        decision="rejected",
                        desc=desc,
                        reason=f"Invalid ImageType values {matches}",
                    )
                )
                continue
        else:
            log.debug(
                msg.format(
                    keyword="ImageType",
                    decision="rejected",
                    desc=desc,
                    reason="ImageType not found",
                )
            )
            continue

        # 3. Reject any series without SliceThickness - we only want image series
        # if thickness is None, series is assumed to be non-image type
        if not ds.get("SliceThickness", None):
            continue

        # 4. Reject any series with BodyPartExamined matching invalid patterns.
        # See contrast_categories.ipynb, also for possibly improving the filtering
        # Accepted body parts are only ["abdomen", "lung", "wholebody"]
        #  FIXME: verify accepted body types,
        if body_part := ds.get("BodyPartExamined", None):
            if body_part.lower() in INVALID_BODY_PART_PATTERN:
                log.debug(
                    msg.format(
                        keyword="BodyPartExamined",
                        decision="rejected",
                        desc=desc,
                        reason=f"Invalid BodyPartExamined: {body_part}",
                    )
                )
                continue
        else:
            # Do not skip to next dataset since BodyPartExamined is optional tag and may be missing.
            # Instead pass to next SeriesDescription filter. If also passed then segmentation will resolve
            # if volume has region of insterest based on vertebrae segmentation
            log.debug(
                msg.format(
                    keyword="BodyPartExamined",
                    decision="accepted",
                    desc=desc,
                    reason="BodyPartExamined not found",
                )
            )

        # FIXME: Decide wether to remove or keep
        # 4. Reject any series with ConvolutionKernel matching lung kernel (Bl57) or Head (Hr*)
        # If kernel == None series is assumed to be non-image type

        # if convolution_kernel := ds.get("ConvolutionKernel"):
        #     convolution_kernel = (
        #         convolution_kernel[0]
        #         if isinstance(convolution_kernel, MultiValue)
        #         else convolution_kernel
        #     ).lower()
        #     if any(k in convolution_kernel for k in ("bl57", "hr")):
        #         continue

        # 5. Reject any series with matching keywords
        # Should remove any remaining secondary image or non-image series.
        if INVALID_SERIES_DESC_PATTERN.search(desc):
            continue

        # all filters passed, add series for C-MOVE request
        uid = ds.get("SeriesInstanceUID")
        datasets_by_uid[uid].append(ds)
    return datasets_by_uid, dose_files
