from pathlib import Path

import imageio
import numpy as np
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from skimage.color import label2rgb

SPINE_COLORS = np.array(
    [
        [0.737, 0.741, 0.133],  # tab:olive
        [0.173, 0.627, 0.173],  # tab:green
        [0.580, 0.403, 0.741],  # tab:purple
        [0.839, 0.153, 0.157],  # tab:red
        [0.122, 0.467, 0.706],  # tab:blue
        [1.000, 0.498, 0.055],  # tab:orange
    ]
)

TISSUE_COLORS = ["khaki", "lightgreen", "dodgerblue", "lightcoral"]
TISSUE_COLORS_2 = np.array(
    [
        [0.965, 0.745, 0.506],
        [0.549, 0.773, 0.520],
        [0.604, 0.530, 0.878],
        [1.000, 0.533, 0.522],
    ]
)

RAS_TO_LPI = np.array([[0.0, -1.0], [1.0, -1.0], [2.0, -1.0]])  # from RAS to LPI


def overlay_spine_mask(
    ct_volume: Nifti1Image,
    spine_mask: Nifti1Image,
    vert_body_centroid: NDArray | list,
    output_dir: Path,
):
    coronal = [
        np.squeeze(
            vol.slicer[:, vert_body_centroid[1] : vert_body_centroid[1] + 1, :]
            .as_reoriented(RAS_TO_LPI)
            .get_fdata(),
            axis=1,
        ).T
        for vol in (ct_volume, spine_mask)
    ]
    sagittal = [
        np.squeeze(
            vol.slicer[vert_body_centroid[0] : vert_body_centroid[0] + 1, :, :]
            .as_reoriented(RAS_TO_LPI)
            .get_fdata(),
            axis=0,
        ).T
        for vol in (ct_volume, spine_mask)
    ]

    coronal[0] = normalize_hu(apply_ct_window(coronal[0], width=1000, level=400))
    sagittal[0] = normalize_hu(apply_ct_window(sagittal[0], width=1000, level=400))

    filenames = ("spine_coronal_overlay.png", "spine_sagittal_overlay.png")
    for (image, mask), filename in zip([coronal, sagittal], filenames):
        try:
            overlay = (
                label2rgb(
                    label=mask,
                    image=image,
                    colors=SPINE_COLORS,
                    alpha=0.5,
                    bg_color=None,
                    bg_label=0,
                )
                * 255
            ).astype(np.uint8)

            fullpath = output_dir.joinpath(filename)
            imageio.imsave(fullpath, overlay)

        except RuntimeError as err:
            print(err)


def overlay_tissue_mask(
    tissue_volume: Nifti1Image,
    tissue_mask: Nifti1Image,
    output_dir: Path,
):
    # reorient slices into LPI direction for 2D plane
    tissue_volume = tissue_volume.as_reoriented(RAS_TO_LPI)
    tissue_mask = tissue_mask.as_reoriented(RAS_TO_LPI)

    # tranpose array for the correct axis directions on 2D plane - right hand on left image side, anterior facing upwards
    image_array, mask_array = [
        np.squeeze(img.get_fdata(), axis=-1).T for img in (tissue_volume, tissue_mask)
    ]

    image_array = normalize_hu(apply_ct_window(image_array, width=600, level=150))

    try:
        overlay = (
            label2rgb(
                label=mask_array,
                image=image_array,
                colors=TISSUE_COLORS,
                alpha=0.8,
                bg_color=None,
                bg_label=0,
            )
            * 255
        ).astype(np.uint8)

        # filename = phase + "_" + "tissue_overlay.png"
        output_filepath = output_dir.joinpath("tissue_overlay.png")

        imageio.imsave(output_filepath, overlay)
    except RuntimeError as err:
        print(err)


def save_preview_image():
    pass


def apply_ct_window(array, width=400, level=50):
    window_range = (level - width // 2, level + width // 2)
    return np.clip(array, window_range[0], window_range[1])


def normalize_hu(array):
    return (array - array.min()) / (array.max() - array.min())
