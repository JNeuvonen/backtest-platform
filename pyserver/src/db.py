import sqlite3
import os
from typing import List

from pandas.io.parquet import json
from constants import DB_DATASETS, DB_WORKER_QUEUE
from db_statements import (
    CREATE_SCRAPE_JOB_TABLE,
    DELETE_SCRAPE_JOB,
    INSERT_SCRAPE_JOB,
    UPDATE_SCRAPE_JOB_TRIES,
)

from load_data import ScrapeJob, load_data
from db_models import RowScrapeJob


def create_connection(db_file: str):
    conn = None
    conn = sqlite3.connect(db_file)
    return conn


def get_tables(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    cursor.close()
    return [table[0] for table in tables]


def create_binance_scrape_job_table(cursor: sqlite3.Cursor):
    cursor.execute(CREATE_SCRAPE_JOB_TABLE)


def poll_scrape_jobs(app_data_path: str):
    cursor = None
    db_worker_queue_conn = create_connection(
        os.path.join(app_data_path, DB_WORKER_QUEUE)
    )
    db_datasets_conn = create_connection(os.path.join(app_data_path, DB_DATASETS))
    try:
        db_worker_queue_conn.row_factory = sqlite3.Row
        cursor = db_worker_queue_conn.cursor()
        statement = "SELECT * FROM binance_scrape_job"
        cursor.execute(statement)

        jobs = []
        ongoing_jobs = False
        for row in cursor.fetchall():
            job = RowScrapeJob(**row)
            if job.ongoing == 1:
                ongoing_jobs = True

            if job.finished == 1 or job.tries == 3:
                cursor.execute(DELETE_SCRAPE_JOB, (job.id,))

            jobs.append(job)

        if ongoing_jobs is False:
            for job in jobs:
                if job.finished == 0:
                    job.tries += 1
                    cursor.execute(
                        UPDATE_SCRAPE_JOB_TRIES,
                        (
                            job.tries,
                            job.id,
                        ),
                    )
                    load_data(
                        app_data_path, job, db_datasets_conn, db_worker_queue_conn
                    )
        db_worker_queue_conn.commit()
        cursor.close()

    except sqlite3.IntegrityError as e:
        print(f"An integrity error occurred: {e}")
        return False
    except sqlite3.ProgrammingError as e:
        print(f"A programming error occurred: {e}")
        return False
    except sqlite3.DatabaseError as e:
        print(f"A database error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    finally:
        if cursor:
            cursor.close()


def insert_binance_scrape_job(conn: sqlite3.Connection, job: ScrapeJob):
    cursor = None
    try:
        cursor = conn.cursor()
        create_binance_scrape_job_table(cursor)

        columns_to_drop_str = json.dumps(job.columns_to_drop)

        cursor.execute(
            INSERT_SCRAPE_JOB,
            (
                job.url,
                job.margin_type,
                job.prefix,
                columns_to_drop_str,
                job.dataseries_interval,
                job.klines,
                job.pair,
                job.candle_interval,
                job.market_type,
                job.path,
                0,
                0,
                0,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        print(f"An integrity error occurred: {e}")
        return False
    except sqlite3.ProgrammingError as e:
        print(f"A programming error occurred: {e}")
        return False
    except sqlite3.DatabaseError as e:
        print(f"A database error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
    finally:
        if cursor:
            cursor.close()

    return True


def get_column_names(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]
    cursor.close()
    return columns
