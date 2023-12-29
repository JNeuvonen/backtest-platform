from constants import DATASET_UTIL_TABLE_NAME, DatasetUtilsColumns


CREATE_DATASET_UTILS_TABLE = f"""
    CREATE TABLE IF NOT EXISTS {DATASET_UTIL_TABLE_NAME} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {DatasetUtilsColumns.TABLE_NAME.value} TEXT NOT NULL,
        {DatasetUtilsColumns.TIMESERIES_COLUMN.value} TEXT NOT NULL
    );
"""
