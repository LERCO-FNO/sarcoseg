import argparse
import sys
from datetime import datetime
from pathlib import Path

from src import preprocessing, segmentation, slogger, utils
from src.classes import StudyData
from src.network import database, pacs


def get_args():
    parser = argparse.ArgumentParser(
        prog="sarcoseg",
        description="segmentation of L3 axial tissues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print more information to console",
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
    parser.add_argument(
        "-pl",
        "--participant-list",
        type=str,
        help="path to list of labkey data (.csv/.txt)",
        required=True,
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

    return parser.parse_args()


def main(args: argparse.Namespace):
    verbose = args.verbose
    main_logger = slogger.get_logger(__name__)

    participant_list = utils.read_patient_list(args.participant_list)

    participant_list = participant_list.participant.to_list()

    labkey_api = database.LabkeyAPI.init_from_json(verbose=verbose)
    if not labkey_api.is_labkey_reachable():
        main_logger.critical("labkey is unreachable")
        sys.exit(-1)

    participant_list = labkey_api.exclude_finished_studies(participant_list)

    queried_study_cases: list[StudyData] = labkey_api._select_rows(
        schema_name="lists",
        query_name="RDG-CT-Sarko-All",
        columns=[
            "ID",
            "PARTICIPANT",
            "RODNE_CISLO",
            "STUDY_INSTANCE_UID",
            "PACS_CISLO",
            "VYSKA_PAC.",
        ],  # TODO: possibly add CAS_VYSETRENI
        filter_dict={"PARTICIPANT": participant_list},
        sanitize_rows=True,
    )

    if not queried_study_cases:
        main_logger.critical(
            "quitting sarcoseg, labkey query response has no StudyInstanceUIDs"
        )
        sys.exit(-1)

    pacs_api = pacs.PacsAPI.init_from_json(verbose=verbose)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir, timestamp)
    output_dir.mkdir(exist_ok=True)

    for study_case in queried_study_cases:
        input_study_dir = Path(args.input_dir, study_case.study_inst_uid)

        if not input_study_dir.exists() and list(input_study_dir.rglob("*")) != 0:
            main_logger.info(
                f"input study directory `{input_study_dir}` not found, trying to download from PACS instead"
            )

            status = pacs_api._movescu(
                study_case.study_inst_uid,
                input_study_dir,
            )

            if status == -1:
                continue

        output_study_dir = Path(output_dir, study_case.study_inst_uid)

        main_logger.info(
            f"preprocessing study {study_case.study_inst_uid} participant {study_case.participant}"
        )
        preprocessing.preprocess_dicom_study(
            input_study_dir,
            output_study_dir,
            study_case,
        )

        if not study_case.series:
            main_logger.warning(
                f"participant {study_case.participant} study {study_case.study_inst_uid} has no series to segment"
            )
            continue

        main_logger.info(
            f"segmenting study {study_case.study_inst_uid} for patient {study_case.participant}"
        )
        segmentation_result = segmentation.segment_ct_study(
            output_study_dir,
            output_study_dir,
            study_case=study_case,
            save_mask_overlays=True,
        )

        print(segmentation_result)

        if args.upload_labkey:
            # TEST:
            labkey_api._upload_data(
                schema_name="lists",
                query_name="CTVysetreni",
                rows=study_case._to_list_of_dicts(),
            )

            # TEST:
            labkey_api._upload_data(
                schema_name="lists",
                query_name="CTSegmentationData",
                rows=segmentation_result._to_list_of_dicts(),
            )

        if args.remove_dicom_files:
            utils.remove_dicom_dir(input_study_dir)

    if not any(output_dir.iterdir()):
        utils.remove_empty_segmentation_dir(output_dir)

    utils.make_report(queried_study_cases, output_dir, timestamp)


if __name__ == "__main__":
    args = get_args()
    main(args)
