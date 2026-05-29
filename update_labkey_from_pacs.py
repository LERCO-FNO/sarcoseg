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

from src.network import database, pacs


def get_args():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--schema",
        type=str,
        help="table type",
        choices=("lists", "dataset"),
        default="lists",
        required=True,
    )
    parent_parser.add_argument("--query", type=str, help="table name", required=True)

    parser = ArgumentParser(
        prog="update_labkey_from_pacs",
        description="This script is intended to update columns in StudyInstanceUID and PACS_CISLO on Labkey for all CT studies.",
    )

    command_parsers = parser.add_subparsers(dest="command", help="labkey APi actions")
    select_parser = command_parsers.add_parser(
        "select", parents=[parent_parser], help="select data from Labkey"
    )
    select_parser.add_argument("--columns", help="column names", nargs="+")

    update_parser = command_parsers.add_parser(
        "update", help="update data on Labkey", parents=[parent_parser]
    )
    update_parser.add_argument(
        "--filepath", help="path to table with data", required=True
    )

    return parser.parse_args()


def select(api: database.LabkeyAPI, schema: str, query: str, columns: list[str]):

    study_root_qr_model_find = _QR_CLASSES.get(
        "StudyRootQueryRetrieveInformationModelFind"
    )
    if not study_root_qr_model_find:
        raise ValueError()

    response = api.query.select_rows(
        schema_name=schema,
        query_name=query,
        columns=",".join(columns),
        max_rows=-1,
        filter_array=[
            QueryFilter("study_instance_uid", "", QueryFilter.Types.IS_NOT_BLANK)
        ],
    )

    if not response.get("rows", None):
        print("no rows returned from labkey")
        sys.exit(-1)

    raw_rows = [
        {key: val for key, val in r.items() if key in columns}
        | {"StudyDescription": ""}
        for r in response.get("rows", None)
    ]
    print(f"returned rows {len(raw_rows)} from {query}")

    if not raw_rows:
        sys.exit(-1)

    pacs_api = pacs.PacsAPI.init_from_json()

    ae = AE(ae_title=pacs_api.aet)
    ae.add_requested_context(study_root_qr_model_find)

    assoc = ae.associate(pacs_api.ip, pacs_api.port, ae_title=pacs_api.aec)
    if not assoc.is_established:
        print("can't establish PACS association")
        sys.exit(-1)

    for row in tqdm(raw_rows, miniters=100, dynamic_miniters=False):
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
        ds.StudyDate = DA(datetime.strptime(labkey_date, "%d-%m-%Y").strftime("%Y%m%d"))
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
            resp.get("StudyInstanceUID", "n/a") for resp in success_resp
        ]
        row["StudyDescription"] = [
            resp.get("StudyDescription", "n/a") for resp in success_resp
        ]
        row["StudyDate"] = [
            datetime.strptime(resp.get("StudyDate", "n/a"), "%Y%m%d").strftime(
                "%d-%m-%Y"
            )
            for resp in success_resp
        ]
        # row["StudyTime"] = [resp.get("StudyTime", "n/a") for resp in success_resp]

    assoc.release()
    if assoc.is_released:
        print("PACS association released")

    single_studies = [r for r in raw_rows if len(r["STUDY_INSTANCE_UID"]) == 1]
    multiple_studies = [r for r in raw_rows if len(r["STUDY_INSTANCE_UID"]) > 1]
    print(f"# of patients with single study in queried date: {len(single_studies)}")
    print(
        f"# of patients with multiple studies in queried date: {len(multiple_studies)}"
    )

    if len(single_studies) > 0:
        df_single_studies = (
            pd.DataFrame(single_studies)
            .explode(
                ["STUDY_INSTANCE_UID", "StudyDescription", "StudyDate", "StudyTime"]
            )
            .reset_index(drop=True)
        )
        df_single_studies.to_csv(
            f"single_studies_{query}.csv", header=True, index=False, sep=","
        )
    else:
        print("no studies in single_studies, writing to csv nothing")

    if len(multiple_studies) > 0:
        df_multiple_studies = (
            pd.DataFrame(multiple_studies)
            .explode(
                ["STUDY_INSTANCE_UID", "StudyDescription", "StudyDate", "StudyTime"]
            )
            .reset_index(drop=True)
        )
        df_multiple_studies.to_csv(
            f"multiple_studies_{query}.csv", header=True, index=False, sep=","
        )
    else:
        print("no studies in multiple_studies, writing to csv nothing")


def update(api: database.LabkeyAPI, schema: str, query: str, filepath: str):

    # FIXME: test detecting delimiters!!
    df = pd.read_csv(
        filepath,
        sep="[,;]",
        header=0,
        index_col=None,
        usecols=("PARTICIPANT", "ID", "STUDY_INSTANCE_UID"),
        engine="python",
    )

    rows = [{col: row[col] for col in df.columns} for _, row in df.iterrows()]
    print(f"updating {query} with {len(rows)} rows")

    response = api.query.update_rows(schema_name=schema, query_name=query, rows=rows)
    print(response)


if __name__ == "__main__":
    args = get_args()
    print(args)

    labkey_api = database.LabkeyAPI.init_from_json()

    schema = args.schema
    query = args.query

    if args.command == "select":
        select(labkey_api, schema, query, args.columns)
    elif args.command == "update":
        update(labkey_api, schema, query, args.filepath)
