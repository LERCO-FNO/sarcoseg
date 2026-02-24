import os
from pathlib import Path

import pandas as pd
from labkey.query import QueryFilter
from pydicom import dcmread
from pynetdicom.apps.findscu import findscu

from src.network import database, pacs

"""
This script is intended to update columns in StudyInstanceUID and PACS_CISLO on Labkey for all CT studies.
"""

columns = [
    "ID",
    "PARTICIPANT",
    "RODNE_CISLO",
    "CAS_VYSETRENI",
    "PACS_CISLO",
    "StudyInstanceUID",
]

labkey_api = database.LabkeyAPI.init_from_json()
response = labkey_api.query.select_rows(
    schema_name="lists",
    query_name="RDG-CT-Sarko-All",
    columns=",".join(columns),
    filter_array=[QueryFilter("pacs_cislo", "", QueryFilter.Types.IS_BLANK)],
)

raw_rows = [
    {key: val for key, val in r.items() if key in columns} | {"StudyDescription": ""}
    for r in response.get("rows", None)
]
print(f"returned rows {len(raw_rows)}")


pacs_api = pacs.PacsAPI.init_from_json()

base_args = [
    pacs_api.ip,
    str(pacs_api.port),
    "-aec",
    pacs_api.aec,
    "-aet",
    pacs_api.aet,
    "-k",
    "QueryRetrieveLevel=STUDY",
    "-k",
    "AccessionNumber",
    "-k",
    "StudyInstanceUID",
    "-k",
    "ModalitiesInStudy=CT",
    "--write",  # writes all incoming responses to files
]

for row in raw_rows:
    # remove all files from previous iteration/run
    [os.remove(f.absolute()) for f in Path("./").glob("rsp*.dcm")]

    id = row["RODNE_CISLO"]
    date = row["CAS_VYSETRENI"].split(" ")[0].replace("-", "")
    patient_args = ["-k", f"PatientID={id}", "-k", f"StudyDate={date}"]
    args = base_args + patient_args

    findscu.main(args)

    rsp_files = list(Path("./").glob("rsp*.dcm"))
    for file in rsp_files:
        ds = dcmread(file)
        row["StudyInstanceUID"] = ds.get("StudyInstanceUID")
        row["StudyDescription"] = ds.get("StudyDescription")


single_studies = [r for r in raw_rows if len(r["StudyInstanceUID"]) == 1]
multiple_studies = [r for r in raw_rows if len(r["StudyInstanceUID"]) > 1]

df_single_studies = pd.DataFrame(single_studies).explode("StudyInstanceUID")
df_multiple_studies = pd.DataFrame(multiple_studies).explode("StudyInstanceUID")

df_single_studies.to_csv("single_studies.csv", header=True, index=False, sep=",")
df_multiple_studies.to_csv("multiple_studies.csv", header=True, index=False, sep=",")

print(f"# of patients with single study in queried date: {len(single_studies)}")
print(f"# of patients with multiple studies in queried date: {len(multiple_studies)}")
