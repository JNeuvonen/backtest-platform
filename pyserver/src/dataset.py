import sqlite3
import pandas as pd
from typing import List

from constants import AppConstants


def get_select_columns_str(columns: List[str]):
    columns_str = ""
    idx = 0
    columns_len = len(columns) - 1
    for item in columns:
        if idx != columns_len:
            columns_str += item + ", "
        else:
            columns_str += item
        idx += 1

    return columns_str


def add_prefix_to_dataset(df, prefix, timeseries_col):
    df.add_prefix(prefix)
    df.rename(columns={prefix + timeseries_col: timeseries_col})


def combine_datasets(
    base_df,
    join_df,
    join_table_name,
    base_join_column,
    join_df_column,
    use_prefix=True,
):
    if use_prefix:
        prefix = join_table_name + "_"
        join_df = join_df.add_prefix(prefix)
        join_df_column = prefix + join_df_column

    merged_df = pd.merge(
        base_df,
        join_df,
        left_on=base_join_column,
        right_on=join_df_column,
        how="left",
    )
    merged_df.drop([join_df_column], axis=1, inplace=True)
    return merged_df


def read_dataset_to_mem(dataset_name: str):
    with sqlite3.connect(AppConstants.DB_DATASETS) as conn:
        query = f"SELECT * FROM {dataset_name}"
        df = pd.read_sql_query(query, conn)
        return df


def read_columns_to_mem(db_path: str, dataset_name: str, columns: List[str]):
    columns_str = get_select_columns_str(columns)
    try:
        with sqlite3.connect(db_path) as conn:
            query = f"SELECT {columns_str} FROM {dataset_name}"
            df = pd.read_sql_query(query, conn)
            return df

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
