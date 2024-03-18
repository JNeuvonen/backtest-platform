import { NullFillStrategy } from "../../utils/constants";

export interface DatasetMetadata {
  columns: string[];
  timeseries_col: string | null;
  price_column: string | null;
  start_date: number;
  end_date: number;
  table_name: string;
}

export interface DatasetUtils {
  dataset_name: string;
  id: number;
  price_column: string;
  target_column: string;
  timeseries_column: string;
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
    data: DatasetModel[];
  };
  status: number;
}

export interface FetchModelByNameRes {
  res: {
    model: DatasetModel;
  };
  status: number;
}

export interface DatasetModel {
  dataset_id: number;
  drop_cols: string[];
  optimizer_and_criterion_code: string;
  model_code: string;
  model_id: number;
  model_name: string;
  null_fill_strat: NullFillStrategy;
  target_col: string;
  validation_split: string;
}

export interface Dataset {
  columns: string[];
  head: Array<number[]>;
  tail: Array<number[]>;
  null_counts: NullCounts;
  row_count: number;
  stats_by_col: StatsByCol;
  timeseries_col: string;
  dataset_name: string;
  target_col: string;
  price_col: string;
  id: number;
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

export interface TrainJob {
  id: number;
  name: string;
  epochs_ran: number;
  num_epochs: number;
  is_training: boolean;
  model_name: string;
  backtest_on_validation_set: boolean;
  save_model_every_epoch: boolean;
  backtest_prices: string;
  backtest_kline_open_times: string;
}

export interface EpochInfo {
  epoch: number;
  id: number;
  train_loss: number;
  val_loss: number;
  val_predictions: string;
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

export interface BacktestBalance {
  cash: number;
  kline_open_time: number;
  portfolio_worth: number;
  position: number;
  prediction: number;
  short_debt: number;
  price: number;
}

export interface BacktestObject {
  open_long_trade_cond: string;
  open_short_trade_cond: string;
  close_long_trade_cond?: string;
  close_short_trade_cond?: string;
  profit_factor: number;
  result_perc: number;
  buy_and_hold_result_net: number;
  buy_and_hold_result_perc: number;
  share_of_winning_trades_perc: number;
  share_of_losing_trades_perc: number;
  id: number;
  name?: string;
  trade_count: number;
  model_weights_id?: number;
  train_job_id?: number;
  dadtaset_id?: number;
  data: BacktestBalance[];
  start_balance: number;
  max_drawdown_perc: number;
  end_balance: number;
}

export interface Trade {
  close_price: number;
  open_price: number;
  close_time: number;
  direction: string;
  id: number;
  net_result: number;
  open_time: number;
  percent_result: number;
  predictions: number[];
  prices: number[];
}

export interface FetchBacktestByIdRes {
  data: BacktestObject;
  trades: Trade[];
}

export interface BacktestsResponse {
  res: {
    data: BacktestObject[];
  };
  status: number;
}

export interface BacktestsByDataset {
  res: {
    data: BacktestObject[];
  };
  status: number;
}
