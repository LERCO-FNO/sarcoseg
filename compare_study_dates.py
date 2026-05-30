import sys

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

    columns = [
        "ID",
        "PARTICIPANT",
        "RODNE_CISLO",
        "CAS_VYSETRENI",
        "PACS_CISLO",
        "STUDY_INSTANCE_UID",
    ]

    response = labkey_api.query.select_rows(
        schema_name="lists",
        query_name="RDG-CT-Sarko-All",
        columns=",".join(columns),
        max_rows=-1,
        filter_array=[
            QueryFilter("STUDY_INSTANCE_UID", "", QueryFilter.Types.IS_NOT_BLANK)
        ],
    )

    if not response.get("rows", None):
        print("no rows returned from labkey")
        sys.exit(-1)

    raw_rows = [
        {key: val for key, val in r.items() if key in columns}
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

    for row in tqdm(raw_rows, miniterval=5, maxinterval=5):
        # labkey_date, labkey_time = row["CAS_VYSETRENI"].split(
        #     " "
        # )  # from format Y-m-d H:M:S
        # labkey_time = labkey_time.split(
        #     ":"
        # )  # from H:M:S into [H, M, S] to filter by hour only

        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.PatientID = ""  # row["RODNE_CISLO"]
        # DA(datetime.strptime(labkey_date, "%Y-%m-%d").strftime("%Y%m%d")),
        ds.StudyDate = ""
        # ds.StudyTime = f"{time_range[0]:02}-{time_range[1]:02}"  # format as 2 digit with leading zero for hours < 10
        ds.StudyTime = ""
        ds.ModalitiesInStudy = "CT"
        ds.PatientSize = ""
        ds.PatientWeight = ""
        ds.AccessionNumber = ""
        ds.AccessionNumber = ""
        ds.StudyInstanceUID = row["STUDY_INSTANCE_UID"]
        ds.StudyDescription = ""

        response = assoc.send_c_find(ds, study_root_qr_model_find)
        success_resp = [msg_id for stat, msg_id in response if stat.Status == 0xFF00]

        row["PACS_PATIENT_ID"] = [resp.get("PatientID") for resp in success_resp][0]
        row["PACS_STUDY_DATETIME"] = [
            f"{datetime.strptime(resp.get('StudyDate'), '%Y%m%d').strftime('%Y-%m-%d')} {datetime.strptime(resp.get('StudyTime'), '%H%M%S').strftime('%H:%M:%s')}"
            for resp in success_resp
        ][0]
        row["ACCESSION_NUMBER"] = [
            resp.get("AccessionNumber") for resp in success_resp
        ][0]
        # row["StudyTime"] = [resp.get("StudyTime", "n/a") for resp in success_resp]

    assoc.release()
    if assoc.is_released:
        print("PACS association released")

    df = pd.DataFrame(raw_rows)
    # .explode(
    #     ["STUDY_INSTANCE_UID_PACS", "STUDY_DESCRIPTION", "CT_STUDY_DATE"]
    # )

    df["CAS_VYSETRENI"] = pd.to_datetime(df["CAS_VYSETRENI"])
    df["PACS_STUDY_DATETIME"] = pd.to_datetime(df["PACS_STUDY_DATETIME"])

    df.to_csv("ct_study_datetimes.csv", header=True, sep=";")
    pprint(df.head(3))


if __name__ == "__main__":
    main()
