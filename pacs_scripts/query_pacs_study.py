import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
from pydicom import Dataset
from pynetdicom import AE, sop_class
from tqdm import tqdm


def main():
    pacs_ip = sys.argv[1]
    pacs_port = sys.argv[2]
    aet = sys.argv[3]
    aec = sys.argv[4]
    csv_path = sys.argv[5]

    study_root_qr_model_find = sop_class._QR_CLASSES.get(
        "StudyRootQueryRetrieveInformationModelFind"
    )
    if not study_root_qr_model_find:
        raise ValueError(
            "No StudyRootQueryRetrieveInformationModelFind in pynetdicom library"
        )

    ikis_data = pl.read_csv(
        csv_path,
        separator=";",
        schema_overrides={
            "RODNE_CISLO": pl.String,
            "PATIENTID": pl.String,
            "PACS_CISLO": pl.String,
        },
        null_values="-",
        comment_prefix="#",
    )
    if ikis_data.is_empty():
        raise ValueError("No data in .csv")

    ikis_data = (
        ikis_data.rename({"RODNE_CISLO": "PATIENT_ID", "PATIENTID": "IKIS_ID"})
        .with_columns(
            IKIS_DATETIME=pl.col("CAS_VYSETRENI").str.to_datetime("%d.%m.%Y %H:%M")
        )
        .with_columns(
            IKIS_STUDY_DATE=pl.col("IKIS_DATETIME").dt.date(),
            IKIS_STUDY_TIME=pl.col("IKIS_DATETIME").dt.time(),
        )
    ).select(
        "PARTICIPANT",
        "PATIENT_ID",
        "IKIS_ID",
        "IKIS_STUDY_DATE",
        "IKIS_STUDY_TIME",
        "NAZEV_VYSETRENI",
        "PACS_CISLO",
    )

    print(ikis_data)

    ae = AE(ae_title=aet)  # pacs_api.aet
    ae.add_requested_context(study_root_qr_model_find)
    assoc = ae.associate(pacs_ip, int(pacs_port), ae_title=aec)
    if not assoc.is_established:
        raise ConnectionError("Failed to establish connection with PACS")

    print("PACS connection establish")

    output_data: list[dict] = []
    for study in tqdm(ikis_data.to_dicts(), mininterval=5.0, maxinterval=5.0):
        # first try with actual PatientID
        study_year = study["IKIS_STUDY_DATE"].year
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.PatientID = study["PATIENT_ID"]
        ds.StudyDate = f"{study_year}0101-{study_year}1231"
        ds.ModalitiesInStudy = "CT"
        ds.StudyTime = ""
        ds.AccessionNumber = ""
        ds.StudyInstanceUID = ""
        ds.StudyDescription = ""
        ds.PatientName = ""

        study["responses"] = []
        resp_datasets = cfind(assoc, ds, study_root_qr_model_find)
        study["responses"].extend(resp_datasets)

        # second try with IKIS ID
        ds.PatientID = study["IKIS_ID"]
        resp_datasets = cfind(assoc, ds, study_root_qr_model_find)
        study["responses"].extend(resp_datasets)

        output_data.append(deepcopy(study))

    # pprint(output_data)

    assoc.release()
    if assoc.is_released:
        print("PACS association released")

    response_dtypes = pl.List(
        pl.Struct(
            {
                "ACCESSION_NUMBER": pl.Utf8,
                "PACS_STUDY_DATE": pl.Utf8,
                "PACS_STUDY_TIME": pl.Utf8,
                "STUDY_DESCRIPTION": pl.Utf8,
                "STUDY_INSTANCE_UID": pl.Utf8,
            }
        )
    )

    # normal .DataFrame() or .from_dicts() fails with "TypeError: nested objects are not allowed"
    # as workaround make Series, then add it to outer dataframe
    outer_records = [
        {k: v for k, v in rec.items() if k != "responses"} for rec in output_data
    ]
    response_data = [rec["responses"] for rec in output_data]
    response_series = pl.Series("responses", response_data, dtype=response_dtypes)
    df_outer = pl.DataFrame(outer_records)
    df = df_outer.with_columns(response_series)
    df = df.explode("responses").unnest("responses")
    print(df)
    df.write_csv("responses.csv", separator=";", null_value="-")


def cfind(assoc, ds, qr_model):
    response = assoc.send_c_find(ds, qr_model)
    success_resp = [msg_id for stat, msg_id in response if stat.Status == 0xFF00]

    default = "-"
    resp_datasets = [
        {
            "PACS_STUDY_DATE": ds.get("StudyDate", default),
            "PACS_STUDY_TIME": ds.get("StudyTime", default),
            "ACCESSION_NUMBER": ds.get("AccessionNumber", default),
            "STUDY_INSTANCE_UID": ds.get("StudyInstanceUID", default),
            "STUDY_DESCRIPTION": ds.get("StudyDescription", default),
            "PATIENT_NAME": ds.get("PatientName"),
        }
        for ds in success_resp
    ]

    return resp_datasets


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "USAGE: python3 query_pacs_study.py <PACS_IP> <PACS_PORT> <CALLING_AET> <CALLED_AET> <CSV_PATH>"
        )
        sys.exit(-1)

    main()
