import sys
from copy import deepcopy
from pathlib import Path
from pprint import pprint

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import UTC, datetime

import polars as pl
from dateutil.relativedelta import relativedelta
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
            "PATIENT_ID": pl.String,
            "PACS_CISLO": pl.String,
        },
        null_values="-",
        comment_prefix="#",
    )
    if ikis_data.is_empty():
        raise ValueError("No data in .csv")

    ikis_data = (
        ikis_data.rename(
            {
                "RODNE_CISLO": "PATIENT_ID",
                "PATIENT_ID": "IKIS_ID",
                "NAZEV_VYSETRENI": "IKIS_STUDY_DESCRIPTION",
                "STUDY_INSTANCE_UID": "IKIS_STUDY_INSTANCE_UID",
                "PACS_CISLO": "IKIS_ACCESSION_NUMBER",
            }
        )
        .with_columns(
            HELPER_INDEX=pl.int_range(0, pl.len()),
            IKIS_DATETIME=pl.col("CAS_VYSETRENI").str.to_datetime("%d.%m.%Y %H:%M"),
        )
        .with_columns(
            IKIS_STUDY_DATE=pl.col("IKIS_DATETIME").dt.date(),
            IKIS_STUDY_TIME=pl.col("IKIS_DATETIME").dt.time(),
        )
    )
    ae = AE(ae_title=aet)  # pacs_api.aet
    ae.add_requested_context(study_root_qr_model_find)
    assoc = ae.associate(pacs_ip, int(pacs_port), ae_title=aec)
    if not assoc.is_established:
        raise ConnectionError("Failed to establish connection with PACS")

    print("PACS connection establish")

    output_data: list[dict] = []
    for study in tqdm(ikis_data.to_dicts(), mininterval=5.0, maxinterval=5.0):
        # first try with actual PatientID
        ikis_study_date = study["IKIS_STUDY_DATE"]
        date_range = f"{(ikis_study_date - relativedelta(years=1)).strftime('%Y%m%d')}-{(ikis_study_date + relativedelta(years=1)).strftime('%Y%m%d')}"

        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.PatientID = study["PATIENT_ID"]
        ds.StudyDate = date_range
        ds.ModalitiesInStudy = "CT"
        ds.StudyTime = ""
        ds.AccessionNumber = ""
        ds.StudyInstanceUID = ""
        ds.StudyDescription = ""
        ds.PatientName = ""

        study["responses"] = []
        resp_datasets = cfind(
            assoc,
            ds,
            study_root_qr_model_find,
            ikis_study_date,
            study["IKIS_STUDY_INSTANCE_UID"],
        )
        study["responses"].extend(resp_datasets)

        # second try with IKIS ID
        ds.PatientID = study["IKIS_ID"]
        resp_datasets = cfind(
            assoc,
            ds,
            study_root_qr_model_find,
            ikis_study_date,
            study["IKIS_STUDY_INSTANCE_UID"],
        )
        study["responses"].extend(resp_datasets)

        output_data.append(deepcopy(study))

    assoc.release()
    if assoc.is_released:
        print("PACS association released")

    response_dtypes = pl.List(
        pl.Struct(
            {
                "PACS_ACCESSION_NUMBER": pl.Utf8,
                "PACS_STUDY_DATE": pl.Date,
                "PACS_STUDY_TIME": pl.Time,
                "PACS_STUDY_DESCRIPTION": pl.Utf8,
                "PACS_STUDY_INSTANCE_UID": pl.Utf8,
                "DAYS_SINCE_IKIS_DATE": pl.Int64,
                "STUDY_UID_MATCH": pl.Boolean,
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
    df = (
        df.explode("responses")
        .unnest("responses")
        .group_by("HELPER_INDEX")
        .agg(pl.all().sort_by("DAYS_SINCE_IKIS_DATE"))
        .sort("HELPER_INDEX")
        .explode(pl.exclude("HELPER_INDEX"))
    )

    # null out every value expect specific columns
    df = df.with_columns(_RN=pl.int_range(0, pl.len()).over("HELPER_INDEX"))
    cols_to_null = [
        c
        for c in df.columns
        if c
        not in (
            "HELPER_INDEX",
            "DAYS_SINCE_IKIS_DATE",
            "_RN",
            "PACS_ACCESSION_NUMBER",
            "PACS_STUDY_DATE",
            "PACS_STUDY_TIME",
            "PACS_STUDY_DESCRIPTION",
            "PACS_STUDY_INSTANCE_UID",
            "STUDY_UID_MATCH",
        )
    ]
    df = df.with_columns(
        [
            pl.when(pl.col("_RN") == 0).then(pl.col(c)).otherwise(None).alias(c)
            for c in cols_to_null
        ]
    ).drop(["_RN", "IKIS_DATETIME"])

    print(df)
    df.write_csv("responses.csv", separator=";", null_value="-")


def cfind(assoc, ds, qr_model, ikis_study_date, ikis_study_uid):
    response = assoc.send_c_find(ds, qr_model)
    success_resp = [msg_id for stat, msg_id in response if stat.Status == 0xFF00]

    default = "-"
    resp_datasets = [
        {
            "PACS_STUDY_DATE": datetime.strptime(ds.get("StudyDate", default), "%Y%m%d")
            .astimezone()
            .date(),
            "PACS_STUDY_TIME": datetime.strptime(
                ds.get("StudyTime", default), "%H%M%S.%f"
            )
            .replace(tzinfo=UTC)
            .time(),
            "PACS_ACCESSION_NUMBER": ds.get("AccessionNumber", default),
            "PACS_STUDY_INSTANCE_UID": ds.get("StudyInstanceUID", default),
            "PACS_STUDY_DESCRIPTION": ds.get("StudyDescription", default),
            "PACS_PATIENT_NAME": ds.get("PatientName"),
            "DAYS_SINCE_IKIS_DATE": abs(
                (
                    ikis_study_date
                    - datetime.strptime(ds.get("StudyDate"), "%Y%m%d")
                    .astimezone()
                    .date()
                ).days
            ),
            "STUDY_UID_MATCH": ds.get("StudyInstanceUID") == ikis_study_uid,
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
