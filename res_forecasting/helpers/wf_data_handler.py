"""Wind Farm Data Handler for managing wind turbine data stored in a DuckDB database."""

import os
import logging
import importlib.resources as pkg_resources
from res_forecasting import data
import duckdb
import pandas as pd
import datetime

logging.basicConfig(
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class WFDataHandler:
    """
    Wind Farm Data Handler for managing wind turbine data stored in a DuckDB database.

    This class provides methods to connect to a DuckDB database, preprocess and ingest CSV files with data,
    and retrieve data as pandas DataFrames for further analysis.

    Note, the process is tested to handle wind turbine data files that follow the naming convention from the Penmanshiel and Kelmarsh wind farm datasets.
    It may require modifications to work with other datasets or naming conventions.
    """

    def __init__(
        self, db_filename: str = "wf_data.duckdb", table_name: str = "turbine_data"
    ) -> None:
        """
        Initialize the WFDataHandler by connecting to the DuckDB database file.

        :param db_filename: Name of the DuckDB database file. Defaults to 'wf_data.duckdb'.
        :type db_filename: str
        :param table_name: Name of the table to use for storing turbine data. Defaults to 'turbine_data'.
        :type table_name: str
        :raises Exception: If connection to the DuckDB database fails.
        """
        self.db_filename: str = db_filename
        self.table_name: str = table_name
        self.connection: duckdb.DuckDBPyConnection | None = None

        try:
            with pkg_resources.path(data, db_filename) as db_path:
                self.connection = duckdb.connect(database=db_path)
                logging.info(f"Connected to DuckDB database: {db_filename}")
        except Exception as e:
            logging.error(f"Failed to connect to DuckDB: {e}")
            raise

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        :return: The instance itself.
        :rtype: WFDataHandler
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and close the DuckDB connection.

        :param exc_type: Exception type.
        :param exc_value: Exception value.
        :param traceback: Traceback object.
        """
        self.close()

    def close(self) -> None:
        """
        Closes the DuckDB connection if it is open.
        Logs any errors encountered during closing.
        """
        if self.connection is not None:
            try:
                self.connection.close()
                logging.info("DuckDB connection closed.")
            except Exception as e:
                logging.error(f"Error closing DuckDB connection: {e}")
            finally:
                self.connection = None

    def _extract_place_and_turbine_id(
        self, filename: str
    ) -> tuple[str | None, int | None]:
        """
        Extract the site name (place) and turbine ID from a filename.

        :param filename: The filename to parse.
        :type filename: str
        :return: The extracted place and turbine ID, or (None, None) if extraction fails.
        :rtype: tuple[str | None, int | None]
        """
        parts = os.path.basename(filename).split("_")
        try:
            if parts[0] == "Turbine" and parts[1] == "Data":
                return parts[2], int(parts[3])
        except IndexError:
            pass
        logging.warning(f"Unexpected filename format: {filename}")
        return None, None

    def _get_ordered_filenames(
        self, folder_path: str, exclude_processed: bool = True
    ) -> list[str]:
        """
        Get an alphabetically ordered list of CSV filenames in the specified folder.

        :param folder_path: Path to the folder containing CSV files.
        :type folder_path: str
        :param exclude_processed: Whether to exclude files starting with 'processed'. Defaults to True.
        :type exclude_processed: bool
        :return: List of ordered CSV filenames.
        :rtype: list[str]
        """
        try:
            filenames = [
                f
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and f.lower().endswith(".csv")
                and (not f.lower().startswith("processed"))
            ]
            filenames.sort()
            return filenames
        except FileNotFoundError:
            logging.error(f"Folder not found: {folder_path}")
            return []
        except Exception as e:
            logging.error(f"Error listing files in folder '{folder_path}': {e}")
            return []

    def ingest_csv_files(
        self, filenames: list[str], normalize_names: bool = True
    ) -> None:
        """
        Ingest a list of CSV files into the DuckDB database table.

        :param filenames: List of CSV file paths to ingest.
        :type filenames: list[str]
        :param normalize_names: Whether to normalize column names. Defaults to True.
        :type normalize_names: bool
        """
        if self.connection is not None:
            try:
                logging.info(f"Ingesting CSV files")

                # Check if table exists
                query = f"""
                            SELECT COUNT(*) > 0 AS exists
                            FROM information_schema.tables 
                            WHERE table_name = '{self.table_name}'
                        """
                table_exists = self.connection.execute(query).fetchone()[0]  # type: ignore

                if not table_exists:  # row_number() OVER () AS id,
                    query = f"""
                                CREATE TABLE {self.table_name} AS
                                SELECT 
                                    *
                                FROM tmp
                            """
                else:
                    query = f"""
                                INSERT INTO {self.table_name}
                                SELECT 
                                    *
                                FROM tmp
                            """

                tmp = self.connection.read_csv(filenames, normalize_names=normalize_names, union_by_name=True, filename=False)  # type: ignore

                self.connection.execute(query)
                logging.info(f"Successfully ingested files")

            except Exception as e:
                logging.error(f"Error ingesting file files: {e}")
        else:
            logging.error(
                f"Connection to the database not established. Cannot ingest files."
            )

    def _preprocess_csv_files(
        self, filenames: list[str], columns: list[str] | None = None, header: int = 9
    ) -> list[str]:
        """
        Preprocess CSV files by selecting specified columns and adding site name and turbine ID.

        :param filenames: List of CSV file paths to preprocess.
        :type filenames: list[str]
        :param columns: Columns to select from each CSV. If None, all columns are used.
        :type columns: list[str] | None
        :param header: Row number to use as the column names. Defaults to 9.
        :type header: int
        :return: List of new filenames for the processed CSV files.
        :rtype: list[str]
        """
        logging.info(f"Preprocessing CSV files")
        new_filenames = list()
        for filename in filenames:
            place, turbine_id = self._extract_place_and_turbine_id(filename)
            if not place or not turbine_id:
                logging.warning(f"Skipping file due to extraction error: {filename}")
                continue
            try:
                df = pd.read_csv(filename, usecols=columns, header=header)
                df["site_name"] = place
                df["turbine_id"] = turbine_id
                new_filenames.append(
                    os.path.join(
                        os.path.dirname(filename),
                        f"processed_{os.path.basename(filename)}",
                    )
                )
                df.to_csv(new_filenames[-1], index=False)
            except Exception as e:
                logging.error(f"Error preprocessing file {filename}: {e}")
        return new_filenames

    def ingest_directory(
        self,
        folder_path: str,
        columns: list[str] | None = None,
        normalize_names: bool = True,
        header: int = 9,
    ) -> None:
        """
        Ingest all CSV files in the specified directory into the DuckDB database.

        :param folder_path: Path to the directory containing CSV files.
        :type folder_path: str
        :param columns: Columns to select from each CSV. If None, all columns are used.
        :type columns: list[str] | None
        :param normalize_names: Whether to normalize column names. Defaults to True.
        :type normalize_names: bool
        :param header: Row number to use as the column names. Defaults to 9.
        :type header: int
        """
        if not os.path.isdir(folder_path):
            logging.error(f"Invalid directory: {folder_path}")
            return

        filenames = self._get_ordered_filenames(folder_path, exclude_processed=True)
        if not filenames:
            logging.warning(f"No CSV files found in directory: {folder_path}")
            return

        filenames = [os.path.join(folder_path, f) for f in filenames]
        logging.info(f"Found {len(filenames)} CSV files to ingest.")
        processed_filenames = self._preprocess_csv_files(
            filenames, columns=columns, header=header
        )
        self.ingest_csv_files(processed_filenames, normalize_names=normalize_names)

    def get_df(
        self,
        site_name: str | None = None,
        turbine_id: int | None = None,
        datetime_start: datetime.datetime | None = None,
        datetime_end: datetime.datetime | None = None,
    ) -> pd.DataFrame | None:
        """
        Retrieve data from the DuckDB table as a pandas DataFrame, optionally filtered by site, turbine, and datetime range.

        :param site_name: Filter by site name. If None, do not filter by site.
        :type site_name: str | None
        :param turbine_id: Filter by turbine ID. If None, do not filter by turbine.
        :type turbine_id: int | None
        :param datetime_start: Filter for records after this datetime (inclusive). If None, no lower bound.
        :type datetime_start: datetime.datetime | None
        :param datetime_end: Filter for records before this datetime (exclusive). If None, no upper bound.
        :type datetime_end: datetime.datetime | None
        :return: The resulting DataFrame, or None if an error occurs or connection is not established.
        :rtype: pd.DataFrame | None
        """
        if self.connection is not None:
            try:
                query = f"SELECT * FROM {self.table_name}"
                if site_name or turbine_id or datetime_start or datetime_end:
                    conditions = []
                    if site_name:
                        conditions.append(f"site_name = '{site_name}'")
                    if turbine_id is not None:
                        conditions.append(f"turbine_id = {turbine_id}")
                    if datetime_start:
                        conditions.append(
                            f"date_and_time >= '{datetime_start.strftime('%Y-%m-%d %H:%M:%S')}'"
                        )
                    if datetime_end:
                        conditions.append(
                            f"date_and_time < '{datetime_end.strftime('%Y-%m-%d %H:%M:%S')}'"
                        )
                    query += " WHERE " + " AND ".join(conditions)
                df = self.connection.execute(query).fetchdf()
                return df
            except Exception as e:
                logging.error(f"Error fetching data from DuckDB: {e}")
                return None
        else:
            logging.error(
                "Connection to the database not established. Cannot fetch data."
            )
            return None


if __name__ == "__main__":

    normalize_names = True
    header = 9
    data_folder = os.path.join(".dump", "data", "wf", "turbine_data")
    columns = [
        "# Date and time",
        "Wind speed (m/s)",
        "Wind speed Sensor 1 (m/s)",
        "Wind speed Sensor 2 (m/s)",
        "Density adjusted wind speed (m/s)",
        "Wind direction (°)",
        "Nacelle position (°)",
        "Power (kW)",
        "Potential power default PC (kW)",
        "Potential power learned PC (kW)",
        "Turbine Power setpoint (kW)",
        "Nacelle ambient temperature (°C)",
        "Ambient temperature (converter) (°C)",
        "Capacity factor",
        "Data Availability",
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
        "Yaw bearing angle (°)",
        "Cable windings from calibration point",
    ]

    with WFDataHandler("wf_data.duckdb") as wf_db_handler:
        # wf_db_handler.ingest_directory(data_folder, columns = columns, normalize_names = normalize_names, header = header)
        df = wf_db_handler.get_df(
            site_name="Penmanshiel",
            turbine_id=1,
            datetime_start=datetime.datetime(2021, 1, 1),
            datetime_end=datetime.datetime(2022, 1, 1),
        )
        if df is not None:
            df.info()
