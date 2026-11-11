import argparse
import sys
from pathlib import Path

from loguru import logger as log

from sarcosegv2.io import read_study_cases_list


def get_args():
    parser = argparse.ArgumentParser(
        prog="sarcoseg",
        description="A vertebrae and muscle/fat tissue segmentation at mid-body level of L3 vertebrae",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Path to input directory with DICOM files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="./output",
        help="Path to output directory",
    )
    parser.add_argument(
        "-s",
        "--study-cases-file",
        # dest="study_case_file",
        type=validate_study_cases_file,
        help="Restrict processing to study cases from this file if study identifiers is found in <input-dir>. Only accepts *.csv and *.json.",
    )
    parser.add_argument(
        "--remove-input-ct-volume",
        action="store_true",
        default=False,
        help="Remove stored `input_ct_volume.nii.gz` files from <output-dir>",
    )
    parser.add_argument(
        "-z",
        "--zip-results",
        choices=("output-dir", "study-dir"),
        default="separate-dir",
        help="Compress results as one archive file or per separate study files",
    )
    parser.add_argument(
        "--remove-nnunet-json",
        action="store_true",
        default=False,
        help="Remove output nnUNet json files from <output-dir>: dataset.json, plans.json, predict_from_raw_data_args.json",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print out at verbose level, including debug information",
    )

    return parser.parse_args()


def validate_study_cases_file(path_str: str) -> Path:
    path = Path(path_str.lower())
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {path}")

    if path.suffix not in {".csv", ".json"}:
        raise argparse.ArgumentTypeError(
            f"Unsupported file extension `{path}`, expected *.csv or *.json"
        )
    return path


def main():
    args = get_args()
    print(args)

    if not args.verbose:
        log.remove()
        log.add(sys.stderr, level="INFO")

    df = read_study_cases_list(args.study_cases_file)
    print(df)


if __name__ == "__main__":
    main()
