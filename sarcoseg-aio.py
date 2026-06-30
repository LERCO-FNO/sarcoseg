import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src import preprocessing, segmentation, utils
from src.classes import ProcessResult, Report, StudyData
from src.network import database, pacs
from src.network.database import FILTER_TYPES, QueryFilter
from src.slogger import setup_logger


def get_args():
    parser = argparse.ArgumentParser(
        prog="sarcoseg",
        description="segmentation of L3 axial tissues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="log debug information to console",
        default=False,
    )
    parser.add_argument(
        "-id",
        "--input-dir",
        type=str,
        help="path to input directory",
        default="./inputs",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        type=str,
        help="path to segmentation output directory",
        default="./outputs",
    )
    # parser.add_argument(
    #     "-pl",
    #     "--participant-list",
    #     type=str,
    #     help="path to list of labkey data (.csv/.txt)",
    #     required=True
    # )
    parser.add_argument(
        "-a",
        "--amount",
        type=int,
        help="number of CT cases to query from labkey and segment",
        default=1,
    )
    parser.add_argument(
        "--upload-labkey",
        action="store_true",
        help="upload preprocessing and segmentation results to labkey",
        default=False,
    )
    parser.add_argument(
        "--remove-dicom-files",
        action="store_true",
        help="remove DICOM files from input directory",
        default=False,
    )
    parser.add_argument(
        "--remove-input-ct-volume",
        action="store_true",
        help="remove all input_ct_volume.nii.gz files in <output-dir>",
        default=False,
    )
    parser.add_argument(
        "--remove-nnunet-jsons",
        action="store_true",
        help="remove nnunet dataset.json, plans.json, predict_from_raw_data_args.json files from <output-dir>",
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    debug = args.debug

    setup_logger(debug)
    log = logging.getLogger("sarcoseg")

    # participants_df = utils.read_patient_list(
    #     args.participant_list,
    #     columns=["PARTICIPANT", "CAS_VYSETRENI"],
    # )

    labkey_api = database.LabkeyAPI.init_from_json(debug=debug)
    if not labkey_api.is_labkey_reachable():
        sys.exit(-1)

    queried_study_cases = labkey_api._select_rows(
        schema_name="lists",
        query_name="CT-Sarko-Select-Segmentation",
        max_rows=args.amount,
        columns=[
            "PARTICIPANT",
            # "RODNE_CISLO",
            "STUDY_UID",
            # "PACS_CISLO",
            "PATIENT_HEIGHT",
            # "VAHA_PAC.",
            # "CAS_VYSETRENI",
        ],
        filter_array=[
            # QueryFilter(
            #     "PARTICIPANT",
            #     ";".join(participants_df["PARTICIPANT"].to_list()),
            #     FILTER_TYPES.EQUALS_ONE_OF,
            # ),
            QueryFilter(
                "SEGMENTATION_FINISHED",
                "",
                FILTER_TYPES.IS_BLANK,
            ),
            QueryFilter(
                "PATIENT_HEIGHT",
                "",
                QueryFilter.Types.IS_NOT_BLANK,
            ),
        ],
    )

    # unfinished_cases = labkey_api.exclude_finished_studies(queried_study_cases)

    if not queried_study_cases:
        log.critical(
            "quitting sarcoseg, labkey query responses have no StudyInstanceUIDs"
        )
        sys.exit(-1)

    # study_cases = [
    #     StudyData._from_labkey_row(case)
    #     for case in unfinished_cases.to_dict(orient="records")
    # ]

    study_cases = [StudyData._from_labkey_row(case) for case in queried_study_cases]

    if len(study_cases) == 0:
        log.error("0 cases to process, exit...")
        sys.exit(-1)

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if output_dir.stem == "outputs":
        output_dir = output_dir.joinpath(timestamp)

    output_dir.mkdir(exist_ok=True, parents=True)

    report = Report(timestamp)

    log.info(f"requested {len(queried_study_cases)} cases for segmentation")
    log.info(f"{len(queried_study_cases) - len(study_cases)} cases already segmented")
    log.info(f"preprocessing and segmenting {len(study_cases)} cases")

    pacs_api = pacs.PacsAPI.init_from_json(debug=debug)

    # --- start study case processing ---
    for study_case in study_cases:
        log.info("-" * 50)

        input_study_dir = Path(args.input_dir, study_case.study_inst_uid)

        if not input_study_dir.exists() and list(input_study_dir.rglob("*")) != 0:
            log.info(
                f"input study directory `{input_study_dir}` not found, trying to download from PACS instead"
            )

            status = pacs_api._movescu(
                study_case.study_inst_uid,
                input_study_dir,
            )

            if status == -1:
                report.add_case(
                    study_case.participant,
                    study_case.study_inst_uid,
                    ProcessResult.MISSING_ON_PACS_OR_LOCAL,
                )
                continue

        output_study_dir = Path(
            output_dir, f"{study_case.participant}_{study_case.study_inst_uid}"
        )

        log.info(
            f"preprocessing case {study_case.participant}, study {study_case.study_inst_uid}"
        )
        preprocessing.preprocess_dicom_study(
            input_study_dir,
            output_study_dir,
            study_case,
        )

        if not study_case.series:
            log.warning(
                f"case {study_case.participant}, study {study_case.study_inst_uid} has no series to segment"
            )
            report.add_case(
                study_case.participant,
                study_case.study_inst_uid,
                ProcessResult.NO_SERIES_TO_SEGMENT,
            )
            continue

        log.info(
            f"segmenting case {study_case.participant}, study {study_case.study_inst_uid}"
        )
        segmentation_result = segmentation.segment_ct_study(
            output_study_dir,
            output_study_dir,
            study_case=study_case,
        )
        log.info(
            f"segmenting finished for {study_case.participant}, study {study_case.study_inst_uid}"
        )

        for series_uid, result in segmentation_result.series_results.items():
            report.add_case(
                study_case.participant,
                study_case.study_inst_uid,
                result.status,
                series_uid,
            )

        if args.upload_labkey:
            study_case_list = study_case._to_list_of_dicts()
            if study_case_list:
                labkey_api._insert_rows(
                    schema_name="lists",
                    query_name="CTVysetreni",
                    rows=study_case_list,
                )
            else:
                log.warning(
                    f"case {study_case.participant}, study {study_case.study_inst_uid} has no DICOM data to send to labkey"
                )

            # remote_image_paths = labkey_api.upload_images(output_study_dir)
            segmentation_result_list = segmentation_result._to_list_of_dicts(
                None  # remote_image_paths
            )
            if segmentation_result_list:
                labkey_api._insert_rows(
                    schema_name="lists",
                    query_name="CTSegmentationData",
                    rows=segmentation_result_list,
                )
            else:
                log.warning(
                    f"case {study_case.participant}, study {study_case.study_inst_uid} has no segmentation data to send to labkey"
                )

            labkey_api._insert_rows(
                "lists",
                "CT-Segmentation-Finished",
                rows=[
                    {
                        "PARTICIPANT": study_case.participant,
                        "STUDY_INSTANCE_UID": study_case.study_inst_uid,
                    }
                ],
            )

        if args.zip_study_dir:
            utils.zip_study_dir(output_study_dir)

        if args.remove_dicom_files:
            utils.remove_dicom_dir(input_study_dir)

    report.write_report(output_dir)

    # cleanup output directory
    if not any(output_dir.iterdir()):
        utils.remove_empty_segmentation_dir(output_dir)

    if args.remove_input_ct_volume:
        utils.remove_input_ct_volume(output_dir)

    if args.remove_nnunet_jsons:
        utils.remove_nnunet_jsons(output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
