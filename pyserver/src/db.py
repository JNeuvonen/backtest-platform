import logging
import sqlite3
import statistics
import pickle
from typing import List
from fastapi import HTTPException


from config import append_app_data_path
from constants import AppConstants, DomEventChannels, NullFillStrategy
from dataset import (
    combine_datasets,
    df_fill_nulls,
    get_col_prefix,
    read_columns_to_mem,
    read_dataset_to_mem,
)
from log import LogExceptionContext, get_logger
from orm import DatasetQuery
from request_types import BodyCreateTrain, BodyModelData


def create_connection(db_file: str):
    conn = sqlite3.connect(db_file)
    return conn


def get_dataset_tables():
    with LogExceptionContext():
        with sqlite3.connect(AppConstants.DB_DATASETS) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            tables_data = []
            for table in tables:
                table_name = table[0]

                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [column_info[1] for column_info in cursor.fetchall()]

                cursor.execute(
                    f"SELECT * FROM {table_name} ORDER BY ROWID ASC LIMIT 1;"
                )
                first_row = cursor.fetchone()
                cursor.execute(
                    f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT 1;"
                )
                last_row = cursor.fetchone()

                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]

                tables_data.append(
                    {
                        "table_name": table_name,
                        "timeseries_col": DatasetQuery.get_timeseries_col(table_name),
                        "columns": columns,
                        "start_date": first_row[0],
                        "end_date": last_row[0],
                        "row_count": row_count,
                    }
                )

            cursor.close()
            return tables_data


def exec_python(code: str):
    with LogExceptionContext():
        exec(code)


def get_col_null_count(
    cursor: sqlite3.Cursor, table_name: str, column_name: str
) -> int:
    query = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL"
    cursor.execute(query)
    return cursor.fetchone()[0]


def rename_table(db_path, old_name, new_name):
    with LogExceptionContext():
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")
            conn.commit()


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


async def add_columns_to_table(
    db_path: str, dataset_name: str, new_cols_arr, null_fill_strat: NullFillStrategy
):
    with LogExceptionContext(notification_duration=60000):
        base_df = read_dataset_to_mem(dataset_name)
        if base_df is None:
            return False
        base_df_timeseries_col = DatasetQuery.get_timeseries_col(dataset_name)
        with sqlite3.connect(db_path) as conn:
            for item in new_cols_arr:
                timeseries_col = DatasetQuery.get_timeseries_col(item.table_name)
                if timeseries_col is None:
                    continue
                df = read_columns_to_mem(
                    db_path, item.table_name, item.columns + [timeseries_col]
                )
                if df is not None:
                    base_df = combine_datasets(
                        base_df,
                        df,
                        item.table_name,
                        base_df_timeseries_col,
                        timeseries_col,
                    )

                    col_prefix = get_col_prefix(item.table_name)
                    for col in item.columns:
                        col_prefixed = col_prefix + col
                        df_fill_nulls(base_df, col_prefixed, null_fill_strat)

            base_df.to_sql(dataset_name, conn, if_exists="replace", index=False)
            logger = get_logger()
            logger.log(
                message=f"Succesfully added new columns to dataset {dataset_name}",
                log_level=logging.INFO,
                display_in_ui=True,
                should_refetch=True,
                notification_duration=5000,
                ui_dom_event=DomEventChannels.REFETCH_ALL_DATASETS.value,
            )


def get_all_tables_and_columns(db_path: str):
    with LogExceptionContext():
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            table_dict = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table[0]})")
                columns = [column[1] for column in cursor.fetchall()]
                table_dict[table[0]] = columns

        return table_dict


def delete_dataset_cols(table_name: str, delete_cols):
    with LogExceptionContext():
        with sqlite3.connect(AppConstants.DB_DATASETS) as conn:
            cursor = conn.cursor()

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [info[1] for info in cursor.fetchall()]

            remaining_columns = [col for col in columns if col not in delete_cols]

            if not remaining_columns:
                raise ValueError("No columns would remain after deletion.")

            remaining_columns_str = ", ".join(remaining_columns)

            new_table_name = f"{table_name}_new"
            cursor.execute(
                f"CREATE TABLE {new_table_name} AS SELECT {remaining_columns_str} FROM {table_name}"
            )
            cursor.execute(f"DROP TABLE {table_name}")
            cursor.execute(f"ALTER TABLE {new_table_name} RENAME TO {table_name}")
            conn.commit()


def get_column_from_dataset(table_name: str, column_name: str):
    with LogExceptionContext():
        with sqlite3.connect(AppConstants.DB_DATASETS) as conn:
            cursor = conn.cursor()
            query = f"SELECT {column_name} FROM {table_name};"
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0] if result is not None else None


def get_dataset_table(table_name: str):
    with LogExceptionContext():
        with sqlite3.connect(AppConstants.DB_DATASETS) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            columns = [row[1] for row in columns_info]
            numerical_columns = [
                row[1]
                for row in columns_info
                if row[2] in ("INTEGER", "REAL", "NUMERIC")
            ]

            head, tail = get_first_last_five_rows(cursor, table_name)
            null_counts = {
                column: get_col_null_count(cursor, table_name, column)
                for column in columns
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
                "timeseries_col": DatasetQuery.get_timeseries_col(table_name),
            }


def exec_sql(db_path: str, sql_statement: str):
    with LogExceptionContext():
        db = create_connection(db_path)
        cursor = db.cursor()
        cursor.execute(sql_statement)
        db.commit()
        db.close()


def get_column_detailed_info(
    table_name: str,
    col_name: str,
    timeseries_col_name: str | None,
):
    with LogExceptionContext():
        with sqlite3.connect(AppConstants.DB_DATASETS) as conn:
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


def rename_column(path: str, table_name: str, old_col_name: str, new_col_name: str):
    with LogExceptionContext():
        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"ALTER TABLE {table_name} RENAME COLUMN {old_col_name} TO {new_col_name}"
            )
            conn.commit()


def get_column_names(conn: sqlite3.Connection, table_name: str) -> List[str]:
    with LogExceptionContext():
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        cursor.close()
        return columns
