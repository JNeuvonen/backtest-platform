import { NullFillStrategy } from "../../utils/constants";

export interface DatasetMetadata {
  columns: string[];
  timeseries_col: string | null;
  start_date: number;
  end_date: number;
  table_name: string;
}

export interface DatasetsResponse {
  res: {
    tables: DatasetMetadata[];
  };
  status: number;
}

export interface BinanceBasicTicker {
  symbol: string;
  price: number;
}

export interface BinanceTickersResponse {
  res: {
    pairs: BinanceBasicTicker[];
  };
  status: number;
}

export interface DatasetResponse {
  res: {
    dataset: Dataset;
  };
  status: number;
}

export interface DatasetModelResponse {
  res: {
    data: DatasetModel;
  };
  status: number;
}

export interface DatasetModel {
  dataset_id: number;
  drop_cols: string[];
  hyper_params_and_optimizer_code: string;
  model: string;
  model_id: number;
  name: string;
  null_fill_strat: NullFillStrategy;
  target_col: string;
  validation_split: number[];
}

export interface Dataset {
  columns: string[];
  head: Array<number[]>;
  tail: Array<number[]>;
  null_counts: NullCounts;
  row_count: number;
  stats_by_col: StatsByCol;
  timeseries_col: string;
}

export interface NullCounts {
  kline_open_time: number;
  open_price: number;
  high_price: number;
  low_price: number;
  close_price: number;
  volume: number;
  quote_asset_volume: number;
  number_of_trades: number;
  taker_buy_base_asset_volume: number;
  taker_buy_quote_asset_volume: number;
}

export interface StatsByCol {
  kline_open_time: StatsCol;
  open_price: StatsCol;
  high_price: StatsCol;
  low_price: StatsCol;
  close_price: StatsCol;
  volume: StatsCol;
  quote_asset_volume: StatsCol;
  number_of_trades: StatsCol;
  taker_buy_base_asset_volume: StatsCol;
  taker_buy_quote_asset_volume: StatsCol;
}

export interface StatsCol {
  mean: number;
  median: number;
  min: number;
  max: number;
  std_dev: number;
}

export interface ColumnResponse {
  res: {
    column: Column;
    timeseries_col: string | null;
  };
  status: number;
}

export interface Column {
  rows: Array<number[]>;
  null_count: number;
  stats?: StatsCol;
  kline_open_time: Array<number[]>;
}
