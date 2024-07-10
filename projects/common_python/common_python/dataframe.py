import pandas as pd


def drop_common_cols_from_join_df(base_df: pd.DataFrame, join_df: pd.DataFrame):
    common_columns = set(base_df.columns).intersection(set(join_df.columns))
    join_df.drop(columns=common_columns, inplace=True)
