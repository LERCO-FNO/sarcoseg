import requests
from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter

from src import slogger
from src.classes import LabkeyRow
from src.utils import read_json

logger = slogger.get_logger(__name__)


class LabkeyAPI(APIWrapper):
    def __init__(
        self,
        domain,
        container_path,
        context_path=None,
        use_ssl=True,
        verify_ssl=True,
        api_key=None,
        disable_csrf=False,
        allow_redirects=False,
        verbose=False,
    ):
        super().__init__(
            domain,
            container_path,
            context_path,
            use_ssl,
            verify_ssl,
            api_key,
            disable_csrf,
            allow_redirects,
        )
        self.verbose = verbose

    def is_labkey_reachable(self):
        hostname = self.server_context.hostname
        try:
            response = requests.get(url=hostname, timeout=5)
            if response.status_code == 200:
                logger.info(f"{hostname} reachable: status {response.status_code}")
            else:
                logger.critical(f"{hostname} is unreachable")
                return False

        except requests.exceptions.RequestException as e:
            logger.critical(f"{hostname} is unreachable")
            logger.critical(f"{e}")
            return False
        return True

    def _select_rows(
        self,
        schema_name: str,
        query_name: str,
        columns: list[str] | str = None,
        max_rows: int = -1,
        filter_dict: dict[str, list[str]] = None,
        sanitize_rows: bool = False,
    ) -> list[LabkeyRow] | None:
        logger.info("labkey query:")
        logger.info(
            f"domain: {self.server_context.hostname}, schema: {schema_name}, query: {query_name}, columns: {columns}"
        )

        filter_array = None
        if filter_dict:
            filter_array = [
                QueryFilter(
                    column,
                    ";".join(values),
                    QueryFilter.Types.EQUALS_ONE_OF,
                )
                for column, values in filter_dict.items()
            ]

        response = self.query.select_rows(
            schema_name=schema_name,
            query_name=query_name,
            columns=",".join(columns) if isinstance(columns, list) else columns,
            max_rows=max_rows,
            filter_array=filter_array,
        )

        rows = response.get("rows", [])
        logger.info(f"returned rows: {len(rows)}")
        if len(rows) == 0:
            logger.warning("no returned rows")
            return None

        if sanitize_rows:
            return self.sanitize_response_data(rows)
        return rows

    def sanitize_response_data(self, rows: list[dict]):
        return [LabkeyRow.from_labkey_dict(row) for row in rows]

    def _upload_data(
        self, schema_name: str, query_name: str, rows: list, update_rows: bool = False
    ):
        logger.info(f"sending {len(rows)} rows")

        response = None
        if update_rows:
            response = self.query.update_rows(
                schema_name=schema_name, query_name=query_name, rows=rows
            )
        else:
            response = self.query.insert_rows(
                schema_name=schema_name, query_name=query_name, rows=rows
            )

        logger.info(response)

    def exclude_finished_studies(self, input_participants: list[str]):
        """Query Labkey `CTSegmentationData` table with input participants and exclude participants with finished segmentation.
        If the queried table has no data, ie empty response, `input_participants` is returned instead.

        Args:
            input_participants (list[str]): List of participants to query.

        Returns:
            participants (list[str]): List of participants excluding participants existing in the queried table.
        """

        logger.info("checking for ")

        rows = self._select_rows(
            schema_name="lists",
            query_name="CTSegmentationData",
            columns=["study_inst_uid", "participant"],
            filter_dict={"PARTICIPANT": input_participants},
            sanitize_rows=True,
        )

        if not rows:
            return input_participants

        finished_studies = set([row["participant"] for row in rows])
        logger.info(
            f"excluding {len(finished_studies)} participants due to existing segmentation results"
        )
        return list(set(input_participants).symmetric_difference(finished_studies))

    @classmethod
    def init_from_json(cls, verbose: bool = False):
        """Initialize Labkey API with configuration values from .env file.

        Args:
            verbose (bool, optional): Verbose printing for the API. Defaults to False.

        Returns:
            api (LabkeyAPI): LabkeyAPI instance.
        """
        conf = read_json("./network.json")["labkey"]

        if verbose:
            logger.info(f"initializing Labkey API with: {conf}")

        if not all(conf.values()):
            logger.error(f"some fields are missing values: {conf}")
            return None

        return cls(conf["domain"], conf["container_path"], verbose=verbose)


"""
def labkey_from_dotenv(verbose: bool = False) -> LabkeyAPI:

    config = dotenv_values()
    return LabkeyAPI(config["domain"], config["container_path"], verbose=verbose)
"""
