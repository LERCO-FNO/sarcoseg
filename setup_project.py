import argparse
import json
from pathlib import Path

# REMOVE in future for nnunetv2 branch
# from zipfile import ZipFile
# import huggingface_hub as hf
from totalsegmentator.python_api import download_pretrained_weights

from src.slogger import get_logger

logger = get_logger(__name__)


def setup_project(
    input_dir: str = "./inputs",
    output_dir: str = "./outputs",
    remove_model_zip: bool = False,
):
    # model_dir = "./models"
    # model_name = "muscle_fat_tissue_0_0_2"
    #
    project_dir = Path().resolve()
    for d in (input_dir, output_dir, "./models"):
        d = Path(project_dir, d)
        if d.exists():
            logger.info(f"directory `{d}` exists")
            continue
        d.mkdir()
        logger.info(f"created directory {d}")
    logger.info("available models for download in src/models.json")

    conf_file = project_dir / "network.json"

    # TODO: improve serialization/deserialization in future
    if not conf_file.exists():
        with open(conf_file, "w") as file:
            conf = {
                "pacs": {
                    "ip": "",
                    "port": -1,
                    "aet": "",
                    "aec": "",
                    "aem": "",
                    "store_port": -1,
                },
                "labkey": {"domain": "", "container_path": ""},
                "test_participant": {
                    "test_study_uid": "",
                },
            }
            json.dump(conf, file, indent=4)
        logger.info(f"created network configuration file `{conf_file}`")
    else:
        logger.warning(f"network configuration file `{conf_file}` already exists")

    # download_models(project_dir)

    # model_path = Path(model_dir, model_name)

    # model_repo = "stanfordmimi/multilevel_muscle_adipose_tissue"
    # print(f"downloading zipped model {model_name} from huggingface repo {model_repo}")

    # model_path.mkdir(exist_ok=True)
    # zip_filepath = hf.hf_hub_download(
    #     repo_id=model_repo, filename="all.zip", local_dir=model_path, repo_type="model"
    # )

    # if not model_path.joinpath("model_final_checkpoint.model").exists():
    #     with ZipFile(zip_filepath) as zipf:
    #         zipf.extractall(model_path)
    #         print(f"unzipped model to {model_path}")

    #     if remove_model_zip:
    #         os.remove(zip_filepath)
    #         print(f"removed zip file {zip_filepath}")
    # else:
    #     print(f"muscle_adipose_tissue model exists at `{model_path}`")

    """
    TotalSegmentator task ids:
    291 - organs
    292 - vertebrae
    293 - cardiac
    294 - muscles
    295 - ribs
    298 - 6mm model
    """

    task_ids = [291, 292, 293, 294, 295, 298]
    logger.info("downloading TotalSegmentator models")
    for tid in task_ids:
        download_pretrained_weights(tid)
    logger.info(f"finished downloading TotalSegmentator models: {task_ids}")


def get_args():
    parser = argparse.ArgumentParser(
        prog="setup_project",
        description="Create input/output/model directories, download models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        help="path to input directory",
        type=str,
        default="./inputs",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="path to output directory",
        type=str,
        default="./outputs",
    )
    parser.add_argument(
        "--remove-model-zip",
        action="store_true",
        help="remove downloaded huggingface model ZIP file",
        default=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    setup_project(
        args.input_dir,
        args.output_dir,
        remove_model_zip=args.remove_model_zip,
    )
