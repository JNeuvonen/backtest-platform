import json
import sqlite3
import pandas as pd
import torch
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from constants import (
    AppConstants,
    NullFillStrategy,
    ScalingStrategy,
)
from log import LogExceptionContext
from query_dataset import DatasetQuery
from query_model import ModelQuery


def get_select_columns_str(columns: List[Optional[str]]):
    valid_columns = [item for item in columns if item is not None]
    columns_str = ", ".join(valid_columns)
    return columns_str


def df_fill_nulls_on_dataframe(df: pd.DataFrame, strategy: NullFillStrategy | None):
    if (
        strategy is NullFillStrategy.NONE.value
        or strategy is None
        or strategy == NullFillStrategy.NONE
    ):
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
        if strategy == NullFillStrategy.ZERO.value or strategy == NullFillStrategy.ZERO:
            df[column].fillna(0, inplace=True)

        elif (
            strategy == NullFillStrategy.MEAN.value or strategy == NullFillStrategy.MEAN
        ):
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)

        elif (
            strategy == NullFillStrategy.CLOSEST.value
            or strategy == NullFillStrategy.CLOSEST
        ):
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


def read_columns_to_mem(db_path: str, dataset_name: str, columns: List[str | None]):
    try:
        columns_str = get_select_columns_str(columns)
        with sqlite3.connect(db_path) as conn:
            query = f"SELECT {columns_str} FROM {dataset_name}"
            df = pd.read_sql_query(query, conn)
            return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def load_data(
    dataset_name: str,
    model_id: int,
    target_column: str,
    null_fill_strategy: NullFillStrategy | None,
    train_val_split: List[int] | None = None,
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD,
    scale_target: bool = False,
):
    model = ModelQuery.fetch_model_by_id(model_id)
    timeseries_col = DatasetQuery.get_timeseries_col(dataset_name)
    price_col = DatasetQuery.get_price_col(dataset_name)
    df = read_dataset_to_mem(dataset_name)
    df_fill_nulls_on_dataframe(df, null_fill_strategy)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how="any", inplace=True)

    scaler = None

    if scaling_strategy == ScalingStrategy.MIN_MAX.value:
        scaler = MinMaxScaler()
    elif scaling_strategy == ScalingStrategy.STANDARD.value:
        scaler = StandardScaler()

    if train_val_split:
        split_start_index = int(len(df) * train_val_split[0] / 100)
        split_end_index = int(len(df) * train_val_split[1] / 100)

        val_df = df.iloc[split_start_index : min(split_end_index + 1, len(df))]
        train_df_1 = df.iloc[:split_start_index]
        train_df_2 = df.iloc[split_end_index + 1 :]
        train_df = pd.concat([train_df_1, train_df_2])

        val_kline_open_times = val_df[timeseries_col].copy()

        if price_col in val_df.columns:
            price_col = val_df[price_col].copy()
        else:
            price_col = None

        if scaler is not None:
            if scale_target:
                train_df.loc[:, train_df.columns] = scaler.fit_transform(
                    train_df[train_df.columns]
                )
                val_df.loc[:, val_df.columns] = scaler.transform(val_df[val_df.columns])
            else:
                train_target_temp = train_df.pop(target_column)
                val_target_temp = val_df.pop(target_column)

                train_df.loc[:, train_df.columns] = scaler.fit_transform(
                    train_df[train_df.columns]
                )
                val_df.loc[:, val_df.columns] = scaler.transform(val_df[val_df.columns])

                train_df[target_column] = train_target_temp
                val_df[target_column] = val_target_temp

                del train_target_temp, val_target_temp

        train_target = train_df.pop(target_column)
        val_target = val_df.pop(target_column)

        drop_cols = json.loads(model.drop_cols_on_train)
        train_df.drop(drop_cols, axis=1, inplace=True)
        val_df.drop(drop_cols, axis=1, inplace=True)

        x_train = torch.Tensor(train_df.values.astype(np.float32))
        y_train = torch.Tensor(
            train_target.to_numpy().reshape(-1, 1).astype(np.float64)
        )
        x_val = torch.Tensor(val_df.values.astype(np.float32))
        y_val = torch.Tensor(val_target.to_numpy().reshape(-1, 1).astype(np.float64))

        return (
            x_train,
            y_train,
            x_val,
            y_val,
            val_kline_open_times,
            price_col,
        )
    else:
        if scaler is not None:
            df[df.columns] = scaler.fit_transform(df[df.columns])

        target = df.pop(target_column)
        x_train = torch.Tensor(df.values.astype(np.float32))
        y_train = torch.Tensor(target.to_numpy().reshape(-1, 1).astype(np.float64))
        return x_train, y_train, None, None, None, None


def read_all_cols_matching_kline_open_times(
    table_name: str, timeseries_col: str, kline_open_times: List[int]
):
    with sqlite3.connect(AppConstants.DB_DATASETS) as conn:
        query = f"SELECT * FROM {table_name} WHERE {timeseries_col} IN ({','.join(['?']*len(kline_open_times))})"
        df = pd.read_sql_query(
            query, conn, params=[str(time) for time in kline_open_times]
        )
        return df
