import sqlite3
import pandas as pd
import torch
import numpy as np
from typing import List

from constants import AppConstants, NullFillStrategy
from log import LogExceptionContext


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


def df_fill_nulls_on_dataframe(df: pd.DataFrame, strategy: NullFillStrategy):
    if strategy is NullFillStrategy.NONE:
        return
    with LogExceptionContext():
        for col in df.columns:
            df_fill_nulls(df, col, strategy)


def df_fill_nulls_on_all_cols(dataset_name: str, strategy: NullFillStrategy):
    if strategy is NullFillStrategy.NONE:
        return
    with LogExceptionContext():
        dataset = read_dataset_to_mem(dataset_name)

        for item in dataset.columns:
            df_fill_nulls(dataset, item, strategy)

        with sqlite3.connect(AppConstants.DB_DATASETS) as conn:
            dataset.to_sql(dataset_name, conn, if_exists="replace", index=False)


def df_fill_nulls(df: pd.DataFrame, column: str, strategy: NullFillStrategy):
    with LogExceptionContext():
        if strategy == NullFillStrategy.ZERO:
            df[column].fillna(0, inplace=True)

        elif strategy == NullFillStrategy.MEAN:
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)

        elif strategy == NullFillStrategy.CLOSEST:
            if df[column].isnull().any():
                ffill = df[column].ffill()
                bfill = df[column].bfill()

                ffill_mask = df[column].ffill().notnull()
                bfill_mask = df[column].bfill().notnull()

                index_series = pd.Series(df.index, index=df.index)

                ffill_dist = (
                    index_series[ffill_mask].reindex_like(df, method="ffill") - df.index
                )
                bfill_dist = df.index - index_series[bfill_mask].reindex_like(
                    df, method="bfill"
                )

                df[column] = np.where(ffill_dist <= bfill_dist, ffill, bfill)


def add_prefix_to_dataset(df, prefix, timeseries_col):
    df.add_prefix(prefix)
    df.rename(columns={prefix + timeseries_col: timeseries_col})


def get_col_prefix(table_name: str):
    return table_name + "_"


def combine_datasets(
    base_df,
    join_df,
    join_table_name,
    base_join_column,
    join_df_column,
    use_prefix=True,
):
    if use_prefix:
        prefix = get_col_prefix(join_table_name)
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


def load_train_data(
    dataset_name: str, target_column: str, null_fill_strategy: NullFillStrategy
):
    df = read_dataset_to_mem(dataset_name)
    df_fill_nulls_on_dataframe(df, null_fill_strategy)
    target = df.pop(target_column)

    x_train = torch.Tensor(df.values.astype(np.float32))
    y_train = torch.Tensor(target.to_numpy().reshape(-1, 1).astype(np.float64))
    return x_train, y_train
