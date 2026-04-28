""""Module for dataset extraction."""
import abc
from abc import ABC
from pathlib import Path

import pandas as pd
import polars as pl
import pickle
from loguru import logger
from sqlalchemy import create_engine, text
import os
import shutil
import subprocess
from typing import Optional

from outboxml import config
from outboxml.core.errors import ConfigError
from outboxml.core.pydantic_models import DataModelConfig
from outboxml.core.utils import FilesNames


class Extractor(ABC):
    """Base interface for extracting data.

    Abstract base class that defines the interface for data extraction.
    Inherited user classes should implement the ``extract_dataset`` method
    which returns a pandas or polars DataFrame.

    :var __connection_config: Internal connection configuration.
    :var load_config_from_env: Whether to load configuration from environment.
    :var connection_config: Connection configuration object.

    .. note::
        Subclasses must implement the :meth:`extract_dataset` method.

    Example::

        class MyExtractor(Extractor):
            def extract_dataset(self) -> pd.DataFrame:
                # Your extraction logic here
                return pd.read_csv('data.csv')
    """

    def __init__(self, *params):
        """Initialization."""
        self.__connection_config = None
        self.load_config_from_env = False
        self.connection_config = None

    @abc.abstractmethod
    def extract_dataset(self) -> pd.DataFrame | pl.DataFrame:
        """Extract dataset.

        :return: Dataset.
        :rtype: pandas.DataFrame, polars.DataFrame
        """
        pass

    def load_config(self, connection_config):
        """Load connection config.

        :param connection_config: Connection config.
        """
        self.connection_config = connection_config

    def __check_object(self, dataset: pd.DataFrame | pl.DataFrame):
        """Check and validate dataset.

        This method can be overridden in subclasses to perform data validation
        and verification.

        :param dataset: Dataset to check.
        :type dataset: pandas.DataFrame or polars.DataFrame
        :return: None
        :rtype: None
        """
        pass


class SimpleExtractor(Extractor):
    """Simple extractor for pre-loaded data.

    This extractor is used when data is already loaded into memory as a
    Pandas' DataFrame or Polars' DataFrame. It does not require any.

    :param data: Pre-loaded dataset.
    :type data: Pandas' DataFrame or Polars' DataFrame
    :param *params: Additional parameters (currently unused).

    Example::

        data = pd.read_csv('my_data.csv')
        extractor = SimpleExtractor(data=data)
        dataset = extractor.extract_dataset()
    """
    def __init__(self, data: pd.DataFrame | pl.DataFrame, *params):
        """Initialization.
        :param data: Dataset.
        :type data: pandas.DataFrame.
        :param params: Parameters.
        """
        super().__init__(*params)
        self.data = data

    def extract_dataset(self) -> pd.DataFrame | pl.DataFrame:
        """Extract dataset.

        :return: Dataset.
        :rtype: pandas.DataFrame
        """
        return self.data


class BaseExtractor(Extractor):
    """Base class for extractor supporting multiple data sources.

    Supports extraction from:
    - CSV files
    - Pickle files
    - Parquet files
    - Databases (PostgreSQL, etc.)

    :param data_config: Data configuration object specifying the data source
        and connection parameters.
    :type data_config: DataModelConfig

    :var __data_config: Internal data configuration object.

    Example::

        from outboxml.core.pydantic_models import DataModelConfig

        config = DataModelConfig(
            source="csv",
            local_name_source="data.csv"
        )
        extractor = BaseExtractor(data_config=config)
        dataset = extractor.extract_dataset()
    """

    def __init__(self, data_config: DataModelConfig):
        """Initialization.

        :param data_config: Object of data config.
        :type data_config: DataModelConfig
        """
        super().__init__()
        self.__data_config = data_config

    def _create_db_trigger_postgre(self, table_name: str):
        engine = create_engine(config.connection_params)

        trigger_sql = f"""
            -- 🔹 Function for DML (data operations)
            CREATE OR REPLACE FUNCTION notify_on_dml()
            RETURNS trigger AS $$
            BEGIN
                PERFORM pg_notify(
                    'table_changes',
                    format('DML event: %s on table %s', TG_OP, TG_TABLE_NAME)
                );
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;
            
            -- Drop old trigger if it exists
            DROP TRIGGER IF EXISTS my_trigger ON "{table_name}";
            
            -- Create new trigger for DML
            CREATE TRIGGER my_trigger
            AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON "{table_name}"
            FOR EACH STATEMENT
            EXECUTE FUNCTION notify_on_dml();
            
            
            -- 🔹 Function for DDL (schema changes)
            CREATE OR REPLACE FUNCTION notify_on_ddl()
            RETURNS event_trigger AS $$
            BEGIN
                PERFORM pg_notify(
                    'table_changes',
                    format('DDL event: %s in schema %s', TG_TAG, current_schema())
                );
            END;
            $$ LANGUAGE plpgsql;
            
            -- Drop event trigger if it exists
            DROP EVENT TRIGGER IF EXISTS ddl_notify;
            
            -- Create event trigger
            CREATE EVENT TRIGGER ddl_notify
            ON ddl_command_end
            WHEN TAG IN ('ALTER TABLE')
            EXECUTE FUNCTION notify_on_ddl();
        """

        with engine.begin() as conn:
            conn.execute(text(trigger_sql))

    def extract_dataset(self) -> pd.DataFrame:
        """Extract dataset.

        :return: Dataset.
        :rtype: pandas.DataFrame
        """
        source = self.__data_config.source

        if source in (FilesNames.csv, FilesNames.pickle, FilesNames.parquet):
            dataset = load_dataset_from_local(data_config=self.__data_config)
            table_name = self.__data_config.table_name_source if self.__data_config.table_name_source is not None else "public." + str(
                Path(self.__data_config.local_name_source).stem)
            try:
                dataset.to_sql(table_name,
                               con=config.connection_params,
                               if_exists='replace')
                self._create_db_trigger_postgre(table_name)

            except Exception as exc:
                logger.error('Loading local file to db error||' + str(exc))

        elif source in (FilesNames.database, FilesNames.hadoop):
            dataset = load_dataset_from_db(data_config=self.__data_config)

        else:
            raise ConfigError(f"Invalid source: {source}")

        self.__check_object(dataset=dataset)

        return dataset

    def __check_object(self, dataset: pd.DataFrame):
        pass


def load_dataset_from_local(data_config: DataModelConfig) -> pd.DataFrame:
    """Load dataset from local csv, pickle or parquet file.

    :param data_config: Object of data config.
    :type data_config: DataModelConfig
    :return: Dataset.
    :rtype: pandas.DataFrame
    """
    logger.info("Load data from local file")
    data = None
    if not data_config.local_name_source:
        logger.error("Invalid local name source")
        raise ConfigError("Invalid local name source")
    params = data_config.extra_params if data_config.extra_params else {}

    try:
        if data_config.source == FilesNames.csv:
            data = pd.read_csv(data_config.local_name_source, **params)
        elif data_config.source == FilesNames.pickle:
            with open(data_config.local_name_source, "rb") as f:
                data = pickle.load(f)
        elif data_config.source == FilesNames.parquet:
            data = pd.read_parquet(data_config.local_name_source, **params)

        if data_config.extra_conditions:
            logger.info('Reading extra conditions||' + str(data_config.extra_conditions))
            data = data.query(data_config.extra_conditions)

    except FileNotFoundError:
        logger.error(f"File {data_config.local_name_source} not found")
        raise f"File {data_config.local_name_source} not found"

    if data is not None and not data.empty:
        logger.debug(f"Data loaded successfully from {data_config.local_name_source}")
        return data

    else:
        logger.debug(f"No data loaded from {data_config.local_name_source}")
        raise f"No Data loaded from {data_config.local_name_source}"


def load_dataset_from_db(data_config: DataModelConfig) -> pd.DataFrame:
    """Load dataset from database.

    :param data_config: Object of data config.
    :type data_config: DataModelConfig
    :return: Dataset.
    :rtype: pandas.DataFrame
    """
    data = None

    if not data_config.table_name_source:
        logger.error("Invalid table name source")
        raise ConfigError("Invalid table name source")

    sql_query = " ".join([
        f"""
            select *
            from {data_config.table_name_source}
        """,
        f"""
            where {data_config.extra_conditions}
        """ if data_config.extra_conditions else ""
    ])

    if data_config.source == FilesNames.database:
        logger.info("Load data from database")
        data = database_to_pandas(sql_query=sql_query)

    if data is not None and not data.empty:
        logger.debug(f"Data loaded successfully from {data_config.table_name_source}")
        return data

    else:
        logger.debug(f"No data loaded from {data_config.table_name_source}")
        raise f"No Data loaded from {data_config.table_name_source}"


def database_to_pandas(sql_query: str) -> pd.DataFrame:
    """Load data from database to pandas dataframe.

    :param sql_query: SQL query.
    :type sql_query: str
    :return: Dataset.
    :rtype: pandas.DataFrame
    """
    data = None

    try:
        engine = config.connection_params
        connect = create_engine(engine)

    except:
        logger.error("Cannot connect to database")
        raise "Cannot connect to database"

    try:
        data = pd.read_sql(sql_query, connect)
    except:
        logger.error("Sql query error")

    return data