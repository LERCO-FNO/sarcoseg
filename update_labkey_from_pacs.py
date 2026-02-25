import sys

import pandas as pd
from labkey.query import QueryFilter
from pydicom import Dataset

from pynetdicom import AE
from pynetdicom.sop_class import _QR_CLASSES

from src.network import database, pacs

study_root_qr_model_find = _QR_CLASSES.get("StudyRootQueryRetrieveInformationModelFind")


"""
This script is intended to update columns in StudyInstanceUID and PACS_CISLO on Labkey for all CT studies.
"""

columns = [
    "ID",
    "PARTICIPANT",
    "RODNE_CISLO",
    "CAS_VYSETRENI",
    "PACS_CISLO",
    "STUDY_INSTANCE_UID",
]

labkey_api = database.LabkeyAPI.init_from_json()
response = labkey_api.query.select_rows(
    schema_name="lists",
    query_name="RDG-CT-Sarko-All",
    columns=",".join(columns),
    filter_array=[QueryFilter("pacs_cislo", "", QueryFilter.Types.IS_BLANK)],
)

response = {
    "rows": [
        {
            "PARTICIPANT": "PAT01",
            "RODNE_CISLO": "15",
            "CAS_VYSETRENI": "2025-06-30 00:00:00",
            "PACS_CISLO": "",
            "STUDY_INSTANCE_UID": "",
        },
        {
            "PARTICIPANT": "PAT02",
            "RODNE_CISLO": "024",
            "CAS_VYSETRENI": "2016-05-11 00:00:00",
            "PACS_CISLO": "",
            "STUDY_INSTANCE_UID": "",
        },
    ]
}

if not response.get("rows", None):
    print("no rows returned from labkey")
    sys.exit(-1)

raw_rows = [
    {key: val for key, val in r.items() if key in columns} | {"StudyDescription": ""}
    for r in response.get("rows", None)
]
print(f"returned rows {len(raw_rows)}")

pacs_api = pacs.PacsAPI.init_from_json()

ae = AE(ae_title=pacs_api.aet)
ae.add_requested_context(study_root_qr_model_find)

assoc = ae.associate(pacs_api.ip, pacs_api.port, ae_title=pacs_api.aec)
if not assoc.is_established:
    print("can't establish PACS association")
    sys.exit(-1)

for row in raw_rows:
    ds = Dataset()
    ds.QueryRetrieveLevel = "STUDY"
    ds.PatientID = row["RODNE_CISLO"]
    ds.StudyDate = row["CAS_VYSETRENI"].split(" ")[0].replace("-", "")
    ds.ModalitiesInStudy = "CT"
    ds.AccessionNumber = ""
    ds.StudyInstanceUID = ""
    ds.StudyDescription = ""

    response = assoc.send_c_find(ds, study_root_qr_model_find)
    success_resp = [msg_id for stat, msg_id in response if stat.Status == 0xFF00]

    row["STUDY_INSTANCE_UID"] = [
        resp.get("StudyInstanceUID", "unknown") for resp in success_resp
    ]
    row["StudyDescription"] = [
        resp.get("StudyDescription", "unknown") for resp in success_resp
    ]

assoc.release()
if assoc.is_released:
    print("PACS association released")

single_studies = [r for r in raw_rows if len(r["STUDY_INSTANCE_UID"]) == 1]
multiple_studies = [r for r in raw_rows if len(r["STUDY_INSTANCE_UID"]) > 1]
print(f"# of patients with single study in queried date: {len(single_studies)}")
print(f"# of patients with multiple studies in queried date: {len(multiple_studies)}")

if len(single_studies) > 0:
    df_single_studies = (
        pd.DataFrame(single_studies)
        .explode(["STUDY_INSTANCE_UID", "StudyDescription"])
        .reset_index(drop=True)
    )
    df_single_studies.to_csv("single_studies.csv", header=True, index=False, sep=",")
else:
    print("no studies in single_studies, writing to csv nothing")

if len(multiple_studies) > 0:
    df_multiple_studies = (
        pd.DataFrame(multiple_studies)
        .explode(["STUDY_INSTANCE_UID", "StudyDescription"])
        .reset_index(drop=True)
    )
    df_multiple_studies.to_csv(
        "multiple_studies.csv", header=True, index=False, sep=","
    )
else:
    print("no studies in multiple_studies, writing to csv nothing")
