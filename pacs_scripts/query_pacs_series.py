import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter
from pydicom import Dataset
from pynetdicom import AE, sop_class
from pynetdicom.sop_class import _QR_CLASSES
from tqdm import tqdm

from src.network import database, pacs


def main():
    labkey_api = database.LabkeyAPI.init_from_json()
    response = labkey_api.query.select_rows(
        "lists",
        "RDG-CT-Sarko-All",
        columns="STUDY_INSTANCE_UID",
        filter_array=[
            QueryFilter("STUDY_INSTANCE_UID", "", QueryFilter.Types.IS_NOT_BLANK),
        ],
    )

    study_uids = response.get("rows", None)
    if not study_uids:
        raise ValueError("No study UIDs returned from labkey")

    study_root_qr_model_find = sop_class._QR_CLASSES.get(
        "StudyRootQueryRetrieveInformationModelFind"
    )
    if not study_root_qr_model_find:
        raise ValueError("No StudyRootQueryRetrieveInformationModelFind")

    raw_rows = [
        {"STUDY_UID": val for key, val in data.items() if key in ["STUDY_INSTANCE_UID"]}
        for data in study_uids
    ]

    print(f"returned {len(raw_rows)} rows")

    if not raw_rows:
        raise ValueError("No rows returned from labkey")

    pacs_api = pacs.PacsAPI.init_from_json()

    ae = AE(ae_title=pacs_api.aet)
    ae.add_requested_context(study_root_qr_model_find)

    assoc = ae.associate(pacs_api.ip, pacs_api.port, ae_title=pacs_api.aec)
    if not assoc.is_established:
        raise ConnectionError("Failed to establish connection with PACS")

    print("PACS association established")

    assoc.release()
    if assoc.is_released:
        print("PACS association released")


if __name__ == "__main__":
    main()