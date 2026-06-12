import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pydicom import Dataset
from pynetdicom import AE, sop_class
from tqdm import tqdm

from src.network import database, pacs


def main():
    labkey_api = database.LabkeyAPI.init_from_json()
    response = labkey_api.query.select_rows(
        "lists",
        "CT-Sarko-Select-Segmentation",
        columns="STUDY_UID",
    )

    raw_rows = response.get("rows", None)
    if not raw_rows:
        raise ValueError("No study UIDs returned from labkey")

    study_root_qr_model_find = sop_class._QR_CLASSES.get(
        "StudyRootQueryRetrieveInformationModelFind"
    )
    if not study_root_qr_model_find:
        raise ValueError("No StudyRootQueryRetrieveInformationModelFind")

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

    series_tags = []
    # series_desc_patterns = re.compile(
    #     r"|".join(
    #         [
    #             "protocol",
    #             "topogram",
    #             "scout",
    #             "patient",
    #             "dose",
    #             "report",
    #             "monitor",
    #             "text",
    #             # "planning",
    #             "mip",
    #             "line",
    #             "distance",
    #             "head",
    #             "coronal",
    #             # "cor",
    #             "sag",
    #             "sagital",
    #             "sagittal",
    #             # "bestdiast",
    #             # "bestsyst",
    #             "thick",
    #             "result",
    #             "bl57",
    #             "vrt",
    #             "view",
    #             "range",
    #             "3d",
    #             "curve",
    #             "vpravo",
    #             "vlevo",
    #             "venozni",
    #             "vytok",
    #             "okno",
    #             "kost",
    #             "snapshot",
    #             "roi",
    #             "circle",
    #             "hlava",
    #             "neck",
    #             "marker",
    #             "private",
    #             "axial",
    #             "regist",
    #             "50kev",
    #             "cad",
    #             "iodine",
    #             "vnc",
    #             "kidney",
    #             "hr",
    #             "ur",
    #             "vessel",
    #         ]
    #     ),
    #     re.IGNORECASE,
    # )

    # series_desc_keep_patterns = re.compile(
    #     r"|".join(
    #         [
    #             "abdomen",
    #             "arterial",
    #             "nephr",
    #             "venous",
    #             "thorax",
    #             "lung",
    #             "angio",
    #             "cta",
    #             "aort",
    #             "chestpain",
    #         ]
    #     ),
    #     re.IGNORECASE,
    # )

    for row in tqdm(raw_rows, mininterval=5.0, maxinterval=5.0):
        ds = Dataset()
        ds.QueryRetrieveLevel = "SERIES"
        ds.StudyInstanceUID = row["STUDY_UID"]
        ds.StudyDescription = ""
        ds.StudyDate = ""
        ds.SeriesDescription = ""
        ds.NumberOfSeriesRelatedInstances = ""
        ds.BodyPartExamined = ""

        response = assoc.send_c_find(ds, study_root_qr_model_find)
        success_resps = [msg_id for stat, msg_id in response if stat.Status == 0xFF00]

        for resp in success_resps:
            series_desc = resp.get("SeriesDescription", "null")

            tags = {
                "study_uid": row["STUDY_UID"],
                "study_desc": resp.get("StudyDescription", None),
                "series_desc": series_desc,
                "series_instances": resp.get("NumberOfSeriesRelatedInstances", None),
                "body_part": resp.get("BodyPartExamined"),
            }

            # early add expected series
            # if series_desc_keep_patterns.search(series_desc):
            #     series_tags.append(tags)
            #     continue

            # skip other series
            # if series_desc_patterns.search(series_desc):
            #     continue

            # add the series if checks above failed
            series_tags.append(tags)

    assoc.release()
    if assoc.is_released:
        print("PACS association released")

    df = pd.DataFrame(series_tags)
    df.head(3)
    print(f"# of rows: {len(df)}")

    df.to_csv("series_tags.csv", index=False, sep=";")


if __name__ == "__main__":
    main()
