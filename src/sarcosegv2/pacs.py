from pathlib import Path
from typing import Any, Self

from loguru import logger as log
from pydicom import Dataset
from pynetdicom import AE, evt
from pynetdicom.association import Association

# pyrefly: ignore[missing-module-attribute]
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelMove

from sarcosegv2.io import read_toml
from sarcosegv2.series_filter import filter_cfind_series_responses

DEFAULT_CONFIG_FILE = Path("./default_config.toml").absolute()
REQUIRED_PACS_PORT_CONFIG_KEYS = ("port", "store_port")
REQUIRED_PACS_STR_CONFIG_KEYS = (
    "ip",
    "calling_ae",
    "called_ae",
    "local_ip",
)


def _handle_store(event, download_dir: Path):
    """
    Store incoming instances under `<download_dir>/<StudyInstanceUID>/<SOPInstanceUID>.dcm`"
    """
    try:
        ds: Dataset = event.dataset
        ds.file_meta = event.file_meta

        study_dir = Path(download_dir) / str(ds.StudyInstanceUID)
        study_dir.mkdir(parents=True, exist_ok=True)
        ds.save_as(study_dir / f"{ds.SOPInstanceUID}.dcm", enforce_file_format=True)
    except Exception as ex:
        log.error(f"{ex}")
        return 0xC001
    return 0x0000


class PacsAccessor:
    _store_handlers = [(evt.EVT_C_STORE, _handle_store)]

    def __init__(
        self,
        ip: str,
        port: int,
        calling_ae: str,
        called_ae: str,
        local_ip: str,
        store_port: int,
        download_dir: str | Path = "./download",
    ):
        self.ip = ip
        self.port = port
        self.calling_ae = calling_ae
        self.called_ae = called_ae
        self.local_ip = local_ip
        self.store_port = store_port
        self.download_dir = Path(download_dir)

    @classmethod
    def from_config(cls, config_file: str | Path | None = None) -> Self:
        config_file = Path(config_file or DEFAULT_CONFIG_FILE)
        log.debug(f"Initializing {cls.__name__} from `{config_file}`")

        config = read_toml(config_file).get("pacs", None)
        if not config:
            raise KeyError(f"Missing [pacs] sectioon in config file: `{config_file}`")

        for k in REQUIRED_PACS_STR_CONFIG_KEYS + REQUIRED_PACS_PORT_CONFIG_KEYS:
            if k not in config:
                raise KeyError(f"`{k}` not found under [pacs] section in config file")

        for k in REQUIRED_PACS_PORT_CONFIG_KEYS:
            port = config.get(k)
            if not 1 <= port <= 65535:
                raise ValueError(f"`{k}` must be between 1 and 65535, got {k}={port}")

        for k in REQUIRED_PACS_STR_CONFIG_KEYS:
            string = config.get(k)
            if not isinstance(string, str) or not string.strip():
                raise ValueError(f"`{k}` must be a non-empty string, got {k}={string}")

        return cls(
            config["ip"],
            config["port"],
            config["calling_ae"],
            config["called_ae"],
            config["local_ip"],
            config["store_port"],
            Path(config["download_directory"]),
        )

    def find_series(self, study_uid: str) -> list[str] | None:
        # pyrefly: ignore[missing-module-attribute]
        from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind

        ds = Dataset()
        ds.StudyInstanceUID = study_uid
        ds.QueryRetrieveLevel = "SERIES"
        ds.SeriesInstanceUID = None
        ds.ImageType = None
        ds.SeriesDescription = None
        ds.BodyPartExamined = None
        ds.SliceThickness = None
        ds.ConvolutionKernel = None

        assoc = self._initialize_association(StudyRootQueryRetrieveInformationModelFind)
        resp_identifiers: dict[str, Dataset] = {}
        if assoc.is_established:
            responses = assoc.send_c_find(
                ds, StudyRootQueryRetrieveInformationModelFind
            )
            for status, identifier in responses:
                if status:
                    log.debug(f"C-FIND query status: 0x{status.Status:04x}")

                    # we only want non-None identifiers (DICOM Dataset)
                    if status.Status == 0xFF00 and identifier:
                        resp_identifiers[identifier.SeriesInstanceUID] = identifier
                        continue

                    if status.Status == 0x0000:  # last response, finished receiving
                        log.debug(f"Final response with status 0x{status.Status:04x}")
                        break

                else:
                    log.error(
                        "Connection timed out, was aborted or received invalid response"
                    )
                    return None
            assoc.release()
        else:
            log.error("Association rejected, aborted or never connected")
            return None

        filtered_series, dose_files = filter_cfind_series_responses(
            resp_identifiers.values()
        )

        series_uids = list(filtered_series.keys())
        if dose_files:
            series_uids.extend([ds.get("SeriesInstanceUID") for ds in dose_files])

        return series_uids

    def move_series(
        self,
        studies: dict[str, list[str]],
        output_directory: str | Path = "./download",
    ):
        """
        Send C-MOVE requests for at QueryRetrieveLevel=SERIES given a map of study_uid: list_of_series_uid and store incoming dataset objects.
        Starts a local SCP to accept incoming objects and store them.

        Parameters
        ----------
        studies : dict(str, list(str))
            Map of study_uid: list_of_series_uid to move.
        output_directory : str | Path, default: "./download"
            Base local storage directory path.

        Returns
        -------
        list(str)
            List of study uids that failed to be moved/stored.

        Raises
        ------
        RuntimeError
            If the local SCP cannot start.
            If association with remote node is not established.
        """
        from pynetdicom import StoragePresentationContexts

        if not self.download_dir and output_directory:
            self.download_dir = Path(output_directory)

        ae = AE(self.calling_ae)
        ae.add_requested_context(StudyRootQueryRetrieveInformationModelMove)
        ae.supported_contexts = StoragePresentationContexts
        handlers = [(evt.EVT_C_STORE, _handle_store, [self.download_dir])]

        scp = ae.start_server(
            (self.local_ip, self.store_port), block=False, evt_handlers=handlers
        )
        if not scp:
            raise RuntimeError(f"Could not create local SCP")

        failures = []

        try:
            assoc = ae.associate(self.ip, self.port, ae_title=self.called_ae)
            if not assoc.is_established:
                raise RuntimeError("Association with remote PACS rejected/failed")

            try:
                for study_uid, series_uids in studies.items():
                    (self.download_dir / study_uid).mkdir(parents=True, exist_ok=True)
                    for series_uid in series_uids:
                        ds = Dataset()
                        ds.QueryRetrieveLevel = "SERIES"
                        ds.StudyInstanceUID = study_uid
                        ds.SeriesInstanceUID = series_uid

                        if not self._move_one_series(assoc, ds):
                            failures.append(study_uid)
            finally:
                assoc.release()
        finally:
            scp.shutdown()
        return failures

    def _move_one_series(self, assoc: Association, ds: Dataset) -> bool:
        responses = assoc.send_c_move(
            ds, self.calling_ae, StudyRootQueryRetrieveInformationModelMove
        )
        ok = False
        for status, identifier in responses:
            if status is None:
                return False
            if status.Status in (0xFF00, 0xFF01):
                continue
            if status.Status == 0x0000:  # finished receiving
                ok = True
            else:
                return False

            if getattr(status, "NumberOfFailedSuboperations", 0):
                ok = False
        return ok

    def is_reachable(self):
        # pyrefly: ignore[missing-module-attribute]
        from pynetdicom.sop_class import Verification

        reachable = False
        assoc = self._initialize_association(Verification)
        if assoc.is_established:
            status = assoc.send_c_echo()
            if status:
                log.debug(f"C-ECHO request status: 0x{status.Status:04x}")
                reachable = True
            else:
                log.error(
                    "Connection timed out, was aborted or received invalid response"
                )
            assoc.release()
        else:
            log.error("Association rejected, aborted or never connected")
        return reachable

    def _initialize_association(self, requested_context: str) -> Association:
        ae = AE(ae_title=self.calling_ae)
        ae.add_requested_context(requested_context)

        return ae.associate(self.ip, self.port, ae_title=self.called_ae)
