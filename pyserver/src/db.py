from enum import Enum
import logging
import os
import sqlite3
import statistics
from typing import List
from log import get_logger


def create_connection(db_file: str):
    conn = sqlite3.connect(db_file)
    return conn


def get_tables(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    tables_data = []
    for table in tables:
        table_name = table[0]

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [column_info[1] for column_info in cursor.fetchall()]

        cursor.execute(f"SELECT * FROM {table_name} ORDER BY ROWID ASC LIMIT 1;")
        first_row = cursor.fetchone()
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT 1;")
        last_row = cursor.fetchone()

        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]

        tables_data.append(
            {
                "table_name": table_name,
                "columns": columns,
                "start_date": first_row[0],
                "end_date": last_row[0],
                "row_count": row_count,
            }
        )

    cursor.close()
    return tables_data


def get_col_null_count(
    cursor: sqlite3.Cursor, table_name: str, column_name: str
) -> int:
    query = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL"
    cursor.execute(query)
    return cursor.fetchone()[0]


def rename_table(db_path, old_name, new_name):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")
            conn.commit()
        return True
    except sqlite3.Error:
        return False


def get_table_row_count(cursor: sqlite3.Cursor, table_name: str) -> int:
    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor.execute(query)
    return cursor.fetchone()[0]


def get_col_stats(cursor: sqlite3.Cursor, table_name: str, column_name: str):
    cursor.execute(
        f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL"
    )
    values = [row[0] for row in cursor.fetchall()]

    if not values:
        return None

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
    }


def get_table_disk_size_bytes(cursor: sqlite3.Cursor, table_name: str) -> int:
    cursor.execute(f"SELECT SUM(pgsize) FROM dbstat WHERE name='{table_name}'")
    size = cursor.fetchone()[0]
    return size if size is not None else 0


def get_first_last_five_rows(cursor: sqlite3.Cursor, table_name: str):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    first_five = cursor.fetchall()
    last_five = []
    if row_count > 5:
        offset = max(0, row_count - 5)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5 OFFSET {offset}")
        last_five = cursor.fetchall()

    return first_five, last_five


def get_dataset_table(conn: sqlite3.Connection, table_name: str):
    logger = get_logger()
    try:
        logger.info("Called get_dataset_table")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        columns = [row[1] for row in columns_info]
        numerical_columns = [
            row[1] for row in columns_info if row[2] in ("INTEGER", "REAL", "NUMERIC")
        ]

        head, tail = get_first_last_five_rows(cursor, table_name)
        null_counts = {
            column: get_col_null_count(cursor, table_name, column) for column in columns
        }
        row_count = get_table_row_count(cursor, table_name)
        numerical_stats = {
            column: get_col_stats(cursor, table_name, column)
            for column in numerical_columns
        }

        return {
            "columns": columns,
            "head": head,
            "tail": tail,
            "null_counts": null_counts,
            "row_count": row_count,
            "stats_by_col": numerical_stats,
        }
    except Exception as e:
        logger.info(f"{e}")
        return None


def exec_sql(db_path: str, sql_statement: str):
    db = create_connection(db_path)
    cursor = db.cursor()
    cursor.execute(sql_statement)
    db.commit()
    db.close()


def get_column_detailed_info(
    conn: sqlite3.Connection,
    table_name: str,
    col_name: str,
    timeseries_col_name: str | None,
):
    logger = get_logger()
    try:
        logger.info("Called get_column_detailed_info")
        cursor = conn.cursor()
        cursor.execute(f"SELECT {col_name} from {table_name}")
        rows = cursor.fetchall()
        null_count = get_col_null_count(cursor, table_name, col_name)
        stats = get_col_stats(cursor, table_name, col_name)

        if timeseries_col_name is not None:
            cursor.execute(f"SELECT {timeseries_col_name} from {table_name}")
            kline_open_time = cursor.fetchall()
        else:
            kline_open_time = None
        return {
            "rows": rows,
            "null_count": null_count,
            "stats": stats,
            "kline_open_time": kline_open_time,
        }

    except Exception as e:
        logger.info(f"{e}")
        return None


def rename_column(path: str, table_name: str, old_col_name: str, new_col_name: str):
    with sqlite3.connect(path) as conn:
        logger = get_logger()
        try:
            logger.info("Called rename_column")
            cursor = conn.cursor()
            cursor.execute(
                f"ALTER TABLE {table_name} RENAME COLUMN {old_col_name} TO {new_col_name}"
            )
            conn.commit()
            logger.info(f"Column renamed from {old_col_name} to {new_col_name}")
        except Exception as e:
            logger.error(f"Error renaming column: {e}")
            return False
        return True


def get_column_names(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]
    cursor.close()
    return columns


class DatasetUtils:
    TABLE_NAME = "dataset_util"
    DB_PATH = "datasets_util.db"

    class Columns(Enum):
        DATASET_NAME = "dataset_name"
        TIMESERIES_COLUMN = "timeseries_column"

    @staticmethod
    def get_path():
        APP_DATA_PATH = os.getenv("APP_DATA_PATH", "")
        return os.path.join(APP_DATA_PATH, DatasetUtils.DB_PATH)

    @staticmethod
    def update_dataset_name(old_name, new_name):
        try:
            with sqlite3.connect(DatasetUtils.get_path()) as conn:
                cursor = conn.cursor()
                update_query = f"""
                    UPDATE {DatasetUtils.TABLE_NAME}
                    SET {DatasetUtils.Columns.DATASET_NAME.value} = ?
                    WHERE {DatasetUtils.Columns.DATASET_NAME.value} = ?
                """
                cursor.execute(update_query, (new_name, old_name))
                conn.commit()
            return True
        except sqlite3.Error:
            return False

    @staticmethod
    def get_timeseries_col(dataset_name: str):
        with sqlite3.connect(DatasetUtils.get_path()) as conn:
            DB_UTIL_COLS = DatasetUtils.Columns
            cursor = conn.cursor()
            query = f"SELECT {DB_UTIL_COLS.TIMESERIES_COLUMN.value} FROM {DatasetUtils.TABLE_NAME} WHERE {DB_UTIL_COLS.DATASET_NAME.value} = ?;"
            cursor.execute(query, (dataset_name,))
            rows = cursor.fetchall()
            cursor.close()
            timeseries_col = None
            try:
                timeseries_col = rows[0][0]
            except Exception as e:
                print(f"An error occured {e}")
            return timeseries_col

    @staticmethod
    def update_timeseries_col(
        dataset_name: str, new_timeseries_col: str | None
    ) -> bool:
        with sqlite3.connect(DatasetUtils.get_path()) as conn:
            col_timeseries = DatasetUtils.Columns.TIMESERIES_COLUMN.value
            col_dataset_name = DatasetUtils.Columns.DATASET_NAME.value
            query = f"""UPDATE {DatasetUtils.TABLE_NAME} SET {col_timeseries} = ?
            WHERE {col_dataset_name} = ?;"""
            try:
                cursor = conn.cursor()
                cursor.execute(query, (new_timeseries_col, dataset_name))
                conn.commit()
                cursor.close()
                return True
            except sqlite3.Error as e:
                print(e)
                return False

    @staticmethod
    async def create_db_utils_entry(dataset_name: str, timeseries_column: str):
        with sqlite3.connect(DatasetUtils.get_path()) as conn:
            DB_UTIL_COLS = DatasetUtils.Columns
            cursor = conn.cursor()
            cursor.execute(
                f"""INSERT INTO {DatasetUtils.TABLE_NAME} ({DB_UTIL_COLS.DATASET_NAME.value}, 
                                {DB_UTIL_COLS.TIMESERIES_COLUMN.value}) VALUES (?, ?)""",
                (
                    dataset_name,
                    timeseries_column,
                ),
            )
            conn.commit()
