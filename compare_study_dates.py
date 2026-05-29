from PIL.ImagePalette import raw
import sys
from argparse import ArgumentParser

import pandas as pd
from labkey.query import QueryFilter
from pydicom import Dataset
from pydicom.valuerep import DA
from pynetdicom import AE
from pynetdicom.sop_class import _QR_CLASSES
from tqdm import tqdm
from datetime import datetime
from pprint import pprint

from src.network import database, pacs


def main():
    labkey_api = database.LabkeyAPI.init_from_json()
    pacs_api = pacs.PacsAPI.init_from_json()

    study_root_qr_model_find = _QR_CLASSES.get(
        "StudyRootQueryRetrieveInformationModelFind"
    )
    if not study_root_qr_model_find:
        raise ValueError()

    columns = ["RODNE_CISLO", "CAS_VYSETRENI", "STUDY_INSTANCE_UID"]

    response = labkey_api.query.select_rows(
        schema_name="lists",
        query_name="RDG-CT-Sarko-All",
        columns=",".join(columns),
        max_rows=100,
        filter_array=[
            QueryFilter("STUDY_INSTANCE_UID", "", QueryFilter.Types.IS_NOT_BLANK)
        ],
    )

    if not response.get("rows", None):
        print("no rows returned from labkey")
        sys.exit(-1)

    raw_rows = [
        {key: val for key, val in r.items() if key in columns}
        | {"STUDY_DESCRIPTION": "", "STUDY_INSTANCE_UID_PACS": []}
        for r in response.get("rows")
    ]

    if not raw_rows:
        sys.exit(-1)

    ae = AE(ae_title=pacs_api.aet)
    ae.add_requested_context(study_root_qr_model_find)

    assoc = ae.associate(pacs_api.ip, pacs_api.port, ae_title=pacs_api.aec)
    if not assoc.is_established:
        print("can't establish PACS association")
        sys.exit(-1)

    for row in tqdm(raw_rows, miniters=100):
        labkey_date, labkey_time = row["CAS_VYSETRENI"].split(
            " "
        )  # from format Y-m-d H:M:S
        labkey_time = labkey_time.split(
            ":"
        )  # from H:M:S into [H, M, S] to filter by hour only

        # extended tiem range to account for a few seconds up to few minute difference between study time on Labkey vs Pacs
        # example, Labkey: 13:59:20.000 vs Pacs: 14:00:40.000
        # need to account for time overflow 23:59 -> 24:00 - must be 00:00 instead

        # time_range = (int(time[0]), upper if (upper := int(time[0]) + 1) < 24 else 0)
        # time_range = f"{int(time[0]):02}-{int(time[0]) + 1:02}"

        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.PatientID = row["RODNE_CISLO"]
        ds.StudyDate = DA(datetime.strptime(labkey_date, "%Y-%m-%d").strftime("%Y%m%d"))
        # ds.StudyTime = f"{time_range[0]:02}-{time_range[1]:02}"  # format as 2 digit with leading zero for hours < 10
        ds.ModalitiesInStudy = "CT"
        ds.PatientSize = ""
        ds.PatientWeight = ""
        ds.AccessionNumber = ""
        ds.StudyInstanceUID = ""
        ds.StudyDescription = ""

        response = assoc.send_c_find(ds, study_root_qr_model_find)
        success_resp = [msg_id for stat, msg_id in response if stat.Status == 0xFF00]

        row["STUDY_INSTANCE_UID_PACS"] = [
            resp.get("StudyInstanceUID") for resp in success_resp
        ]
        row["STUDY_DESCRIPTION"] = [
            resp.get("StudyDescription") for resp in success_resp
        ]
        row["CT_STUDY_DATE"] = [
            datetime.strptime(resp.get("StudyDate"), "%Y%m%d").strftime("%Y-%m-%d")
            for resp in success_resp
        ]
        # row["StudyTime"] = [resp.get("StudyTime", "n/a") for resp in success_resp]

    assoc.release()
    if assoc.is_released:
        print("PACS association released")

    df = pd.DataFrame(raw_rows).explode(
        ["STUDY_INSTANCE_UID_PACS", "STUDY_DESCRIPTION", "CT_STUDY_DATE"]
    )

    df["CAS_VYSETRENI"] = pd.to_datetime(df["CAS_VYSETRENI"])
    df["CT_STUDY_DATE"] = pd.to_datetime(df["CT_STUDY_DATE"])

    df["STUDY_TIME_MATCH"] = df["CAS_VYSETRENI"].dt.date == df["CT_STUDY_DATE"].dt.date
    df["STUDY_UID_MATCH"] = df["STUDY_INSTANCE_UID"] == df["STUDY_INSTANCE_UID_PACS"]

    df.to_csv("single_studies.csv", header=True, sep=";")
    pprint(df.head(3))


if __name__ == "__main__":
    main()
