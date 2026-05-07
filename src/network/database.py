import logging
from typing import Any, Self

import pandas as pd
import requests
from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter

from src.io import read_json

log = logging.getLogger("database")


FILTER_TYPES = QueryFilter.Types


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

    def is_labkey_reachable(self):
        hostname = self.server_context.hostname
        reachable = False

        try:
            response = requests.get(url=hostname, timeout=5)
            if response.status_code == 200:
                reachable = True
            log.debug(f"{response.status_code}")
        except requests.exceptions.RequestException as e:
            log.critical(f"{e.response}")

        log.info(f"{hostname} reachable: {reachable}")
        return reachable

    def _select_rows(
        self,
        schema_name: str,
        query_name: str,
        columns: list[str] | str | None = None,
        max_rows: int = -1,
        filter_array: list[QueryFilter] | None = None,
    ) -> list[dict[str, Any]]:
        log.info(
            f"labkey query, schema: {schema_name}, query: {query_name}, columns: {columns}"
        )

        response = self.query.select_rows(
            schema_name=schema_name,
            query_name=query_name,
            columns=",".join(columns) if isinstance(columns, list) else columns,
            max_rows=max_rows,
            filter_array=filter_array,
        )

        rows = response.get("rows", [])
        log.info(f"SELECT {query_name}: {len(rows)} rows")
        if len(rows) == 0:
            log.warning(f"no returned rows from {query_name}")
            return []

        cols = [col["dataIndex"] for col in response.get("columnModel", [])]
        return [
            {key: value for key, value in row.items() if key in cols} for row in rows
        ]

    def _insert_rows(
        self, schema_name: str, query_name: str, rows: list[dict[str, Any]]
    ):
        response = self.query.insert_rows(
            schema_name=schema_name, query_name=query_name, rows=rows
        )
        log.info(f"INSERT {query_name}: {response.get('rowsAffected', 0)} rows")

    def exclude_finished_studies(self, cases: list[dict[str, Any]]) -> pd.DataFrame:
        """Query Labkey `CTSegmentationData` table with list of Study Instance UIDs and exclude those with segmentation results.
        Returns input list if the queried table has no data for matching values, ie no values for "rows" key.

        Args:
            study_uids (list[str]): List of Study Instance UIDs values to match.

        Returns:
            finished_study_uids (list[str]): List of Study Instance UIDs, excluding those with existing segmentation results.
        """

        log.info("checking for study uids with segmentation results")

        cols = list(cases[0].keys())
        cases_df = pd.DataFrame(cases, columns=cols)

        study_uids = cases_df["STUDY_INSTANCE_UID"].to_list()

        rows = self._select_rows(
            schema_name="lists",
            query_name="CT-Segmentation-Finished",
            columns=["STUDY_INSTANCE_UID"],
            filter_array=[
                QueryFilter(
                    "STUDY_INSTANCE_UID",
                    ";".join(study_uids),
                    FILTER_TYPES.EQUALS_ONE_OF,
                )
            ],
        )

        if not rows:
            log.debug("no study uids excluded, returning input")
            return cases_df

        finished_studies = set([row["STUDY_INSTANCE_UID"] for row in rows])

        log.debug(
            f"excluding {len(finished_studies)} study uids due to existing segmentation results"
        )

        return cases_df[
            ~cases_df["STUDY_INSTANCE_UID"]
            .isin(finished_studies)
            .reset_index(drop=True)
        ]

    @classmethod
    def init_from_json(cls, debug: bool = False) -> Self:
        """Initialize Labkey API with configuration values from .env file.

        Args:
            debug (bool, optional): Debug printing for the API. Defaults to False.

        Returns:
            api (LabkeyAPI): LabkeyAPI instance.
        """
        conf = read_json("./src/network/network.json")["labkey"]

        log.debug(f"initializing Labkey API with: {conf}")

        if not all(conf.values()):
            log.error(f"some fields are missing values: {conf}")
            raise ValueError("Unable to initialize LabkeyAPI")

        return cls(conf["domain"], conf["container_path"])
