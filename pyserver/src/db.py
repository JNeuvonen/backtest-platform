import logging
import sqlite3
import statistics
from typing import List
from log import get_logger


def create_connection(db_file: str):
    conn = None
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


def get_column_detailed_info(conn: sqlite3.Connection, table_name: str, col_name: str):
    logger = get_logger()
    try:
        logger.info("Called get_column_detailed_info")
        cursor = conn.cursor()
        cursor.execute(f"SELECT {col_name} from {table_name}")
        rows = cursor.fetchall()
        null_count = get_col_null_count(cursor, table_name, col_name)
        stats = get_col_stats(cursor, table_name, col_name)
        cursor.execute(f"SELECT kline_open_time from {table_name}")
        kline_open_time = cursor.fetchall()
        return {
            "rows": rows,
            "null_count": null_count,
            "stats": stats,
            "kline_open_time": kline_open_time,
        }

    except Exception as e:
        logger.info(f"{e}")
        return None


async def rename_column(
    conn: sqlite3.Connection, table_name: str, old_col_name: str, new_col_name: str
):
    logger = get_logger()
    await logger.log(
        f"Renamed column on table: {table_name} from {old_col_name} to {new_col_name}",
        logging.INFO,
    )
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


def get_columns_by_table(conn: sqlite3.Connection):
    logger = get_logger()
    try:
        logger.info("Called get_columns_by_table")
        cursor = conn.cursor()
        return {}

    except Exception as e:
        logger.info(f"{e}")
        return None


def get_column_names(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]
    cursor.close()
    return columns
