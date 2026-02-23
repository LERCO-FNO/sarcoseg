import unittest
from pathlib import Path

from src.classes import StudyData
from src.network import database, pacs
from src.io import read_json


class TestPacs(unittest.TestCase):
    api = pacs.PacsAPI.init_from_json()
    STUDY_INST_UID: str

    def test_echoscu(self):
        ret = self.api._echoscu(verbose=True)
        self.assertEqual(ret.returncode, 0)

    def test_movescu(self):
        self.STUDY_INST_UID = read_json("./network.json")["test_participant"].get(
            "test_study_uid"
        )
        validate_study_uid(self.STUDY_INST_UID)

        if not self.STUDY_INST_UID:
            self.fail(
                f"test_movescu requires a valid StudyInstanceUID, given UID is `{self.STUDY_INST_UID}`"
            )

        download_dir = Path(f"./tests/download/{self.STUDY_INST_UID}")

        if not download_dir.exists() and list(download_dir.rglob("*")) != 0:
            print(
                f"input study directory {self.STUDY_INST_UID} not found, trying to download from PACS instead"
            )
            self.api._movescu(self.STUDY_INST_UID, download_dir)

        files = list(download_dir.rglob("*"))

        self.assertNotEqual(
            len(files), 0, "output download directory empty, no DICOM files downloaded"
        )


class TestLabkey(unittest.TestCase):
    api = database.LabkeyAPI.init_from_json(verbose=True)

    def test_labkey_connection(self):
        self.assertTrue(self.api.is_labkey_reachable())

    def test_labkey_query(self):
        rows = [
            {
                "RODNE_CISLO": "0124",
                "VYSKA_PAC.": 179.0,
                "PARTICIPANT": "PT001",
                "STUDY_INSTANCE_UID": "1.2.3",
            }
        ]
        labkey_data = self.api.sanitize_response_data(rows)
        true_labkey_data = [
            StudyData(
                participant="PT001",
                study_inst_uid="1.2.3",
                patient_height=179.0,
            )
        ]

        for data, true_data in zip(labkey_data, true_labkey_data):
            self.assertEqual(data, true_data)
            print(f"{data=}\n{true_data=}")


def validate_study_uid(study_uid: str):
    if "." not in study_uid:
        raise ValueError(
            f"INVALID UID FORMAT: StudyInstanceUID is missing periods `{study_uid}`"
        )
    elif study_uid.islower() or study_uid.isupper():
        raise ValueError(
            f"INVALID UID FORMAT: StudyInstanceUID contains alphabet characters: `{study_uid}`"
        )
    return study_uid


if __name__ == "__main__":
    unittest.main()
