import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Self

from pydicom import dcmread
from pydicom.multival import MultiValue
from pynetdicom.apps.echoscu import echoscu

from src.io import read_json
from src.utils import SERIES_DESC_PATTERN

log = logging.getLogger("pacs")

WRONG_IMAGE_TYPES = ["OTHER", "LOCALIZER", "SECONDARY"]  # "DERIVED", "SECONDARY"


class PacsAPI:
    def __init__(
        self,
        ip: str,
        port: int,
        aet: str,
        aec: str,
        store_port: int,
        aem: str | None = None,
    ):
        self.ip = ip
        self.port = port
        self.aet = aet
        self.aec = aec
        self.aem = aem if aem else aet
        self.store_port = store_port

    def _movescu(self, study_inst_uid: str, download_directory: str | Path):
        response_dir = Path(download_directory, "rsp")
        response_dir.mkdir(exist_ok=True, parents=True)

        args_findscu = [
            "./dcmtk/findscu",
            self.ip,
            str(self.port),
            "-aet",
            self.aet,
            "-aec",
            self.aec,
            "-S",
            "-X",  # store responses
            "-k",
            "QueryRetrieveLevel=SERIES",
            "-k",
            "SeriesDescription",
            "-k",
            "ImageType",
            "-k",
            "ConvolutionKernel",
            "-k",
            f"StudyInstanceUID={study_inst_uid}",
            "-od",
            str(response_dir),
        ]
        result = subprocess.run(args_findscu, capture_output=True, text=True)
        if result.returncode == -1:
            return result.returncode

        response_files = Path(response_dir).glob("rsp*.dcm")

        query_files = []
        for rsp in response_files:
            ds = dcmread(rsp)

            series_desc = ds.get("SeriesDescription", "").lower()

            if "dose report" in series_desc:
                query_files.append(rsp)
                continue

            image_type = ds.get("ImageType", [])
            if image_type and any(
                img_type in WRONG_IMAGE_TYPES for img_type in image_type
            ):
                continue

            if convolution_kernel := ds.get("ConvolutionKernel", ""):
                convolution_kernel = (
                    convolution_kernel[0]
                    if isinstance(convolution_kernel, MultiValue)
                    else convolution_kernel
                )
                if "bl57" in convolution_kernel.lower():
                    continue

            if SERIES_DESC_PATTERN.search(series_desc):
                continue

            query_files.append(rsp)

        args = [
            "./dcmtk/movescu",
            self.ip,
            str(self.port),
            "-aec",
            self.aec,
            "-aet",
            self.aet,
            "--port",
            str(self.store_port),
            "-od",
            str(download_directory),
            "-S",
            "-k",
            "QueryRetrieveLevel=SERIES",
            "-k",
            f"StudyInstanceUID={study_inst_uid}",
            # TODO: add response files here!!!
        ] + query_files

        log.debug(f"running C-MOVE for StudyInstanceUID: {study_inst_uid}")
        result = subprocess.run(args, capture_output=True, text=True)

        # clean the response directory
        shutil.rmtree(response_dir)

        if len(os.listdir(download_directory)) == 0:
            os.rmdir(download_directory)
            log.error(
                f"C-MOVE failed downloading data for StudyInstanceUID: {study_inst_uid}"
            )
            return -1

        log.debug("finished C-MOVE")
        return 0

    def _echoscu(self, debug: bool = False):
        args = [
            sys.executable,
            echoscu.__file__,
            self.ip,
            str(self.port),
            "-aec",
            self.aec,
            "-aet",
            self.aet,
        ]

        if debug:
            args.append("-d")

        ret = subprocess.run(args, capture_output=True, text=True)
        log.info(f"ECHOSCU return code: {ret.returncode}")

        if ret.stdout:
            log.debug(f"ECHOSCU stdout: {ret.stdout}")
        if ret.stderr:
            log.debug(f"ECHOSCU stderr: {ret.stderr}")
        return ret.returncode

    @classmethod
    def init_from_json(cls, debug: bool = False) -> Self:
        conf = read_json("./src/network/network.json")["pacs"]
        log.debug(f"initializing PACS API with: {conf}")

        if not all(conf.values()):
            log.error(f"some fields are missing values: {conf}")
            raise ValueError("Unable to initialize PACS API")

        return cls(
            conf["ip"],
            int(conf["port"]),
            conf["aet"],
            conf["aec"],
            int(conf["store_port"]),
        )


if __name__ == "__main__":
    pacs = PacsAPI.init_from_json()
    pacs._echoscu()
