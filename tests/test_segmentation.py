import unittest
from pathlib import Path

import numpy as np

from src import segmentation, utils, visualization
from src.classes import Centroids, MetricsData, SegmentationResult


class TestSegmentation(unittest.TestCase):
    def test_study_segment(self):
        test_uid = "1.3.6.1.4.1.36302.1.1.2.67388.4692994"
        path = Path("tests/output/", test_uid)
        segmentation.segment_ct_study(
            input_dir=path,
            output_dir=path,
            save_mask_overlays=True,
        )

    def test_spine_segment(self):
        pass

    def test_tissue_segment(self):
        pass

    def test_tissue_extract(self):
        pass

    def test_postproc_tissue_mask(self):
        pass

    def test_compute_metrics(self):
        tissue_volume = utils.read_volume(
            "tests\\output\\1.3.6.1.4.1.36302.1.1.2.67386.4681372\\1.3.12.2.1107.5.1.4.75968.30000025063005325811100017182\\tissue_slices.nii.gz"
        )
        tissue_mask = utils.read_volume(
            "tests\\output\\1.3.6.1.4.1.36302.1.1.2.67386.4681372\\1.3.12.2.1107.5.1.4.75968.30000025063005325811100017182\\tissue_mask.nii.gz"
        )

        metrics = utils.compute_metrics(tissue_mask, tissue_volume, 170.0)
        print(metrics)

    def test_mask_overlays(self):
        pass

    def test_metrics_patient_data(self):
        output_dir = Path("tests/output/1.3.6.1.4.1.36302.1.1.2.67388.4692994")

        seg_result = SegmentationResult(
            participant="PAT1", study_inst_uid=output_dir.name, patient_height=170.0
        )

        all_metrics = {
            "1.2.1": MetricsData(
                area={"sat": 200.0, "muscle": 210.0},
                mean_hu={"sat": -97.0},
                skelet_muscle_index=28.0,
                series_inst_uid="1.2.1",
                contrast_phase="venous",
                duration=35.0,
                centroids=Centroids(
                    vertebre_centroid=np.array([0, 0, 417]).tolist(),
                    body_centroid=np.array([0, 0, 422]).tolist(),
                ),
            ),
            "1.2.2": MetricsData(
                area={"sat": 100.0},
                mean_hu={"sat": -80.0},
                skelet_muscle_index=40.0,
                series_inst_uid="1.2.2",
                contrast_phase="arterial",
                duration=40.0,
                centroids=Centroids(
                    vertebre_centroid=np.array([0, 0, 500]).tolist(),
                    body_centroid=np.array([0, 0, 414]).tolist(),
                ),
            ),
        }
        seg_result.metrics_dict = all_metrics

        seg_result._write_to_json(output_dir)


class MaskOverlay(unittest.TestCase):
    def test_overlay_spine(self):
        pass

    def test_overlay_tissue(self):
        pass


if __name__ == "__main__":
    unittest.main()
