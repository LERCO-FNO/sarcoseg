import json
import unittest
from pathlib import Path

from src import preprocessing
from src.classes import SeriesData, StudyData


class TestPreprocessDicom(unittest.TestCase):
    def test_find_dicom(self):
        path = Path("tests", "input")
        files = preprocessing.find_dicoms(
            path.joinpath("1.3.6.1.4.1.36302.1.1.2.67386.4681372")
        )
        no_files = preprocessing.find_dicoms(path.joinpath("empty"))

        self.assertTrue(files)
        self.assertIsNone(no_files)

    def test_dose_report(self):
        filepath = Path("tests/input/dicom/dose", "00000ADD")

        dose_values = preprocessing.extract_dose_values(filepath)

    def test_study_preprocess(self):
        true_uid = "1.3.6.1.4.1.36302.1.1.2.67388.4692994"
        true_participant = "PAT01"

        input_path = Path("tests/input/", true_uid)
        output_path = Path("tests/output/", true_uid)
        row = StudyData(true_participant, study_inst_uid=true_uid, patient_height=170.0)

        preprocessing.preprocess_dicom_study(input_path, output_path, study_case=row)

        test_study_data = StudyData._from_json(
            output_path.joinpath(f"dicom_tags_{true_uid}.json")
        )
        self.assertEqual(test_study_data.participant, true_participant)
        self.assertEqual(test_study_data.study_inst_uid, true_uid)

        nifti_files = list(output_path.rglob("input_ct_volume.nii.gz"))
        self.assertIsNotNone(nifti_files)

        json_file = list(output_path.rglob("*.json"))
        self.assertIsNotNone(json_file)

        with open(json_file[0], mode="r") as file:
            data = json.load(file)
            series_dict = data.pop("series")
            test_study_data = StudyData(
                **data,
                series={key: SeriesData(**val) for key, val in series_dict.items()},
            )

        print(
            f"participant {test_study_data.participant}, study inst uid {test_study_data.study_inst_uid}"
        )
        self.assertEqual(true_participant, test_study_data.participant)
        self.assertEqual(true_uid, test_study_data.study_inst_uid)


if __name__ == "__main__":
    unittest.main()
