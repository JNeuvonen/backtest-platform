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
  data_transformations: DataTransformation[];
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
    price_col: string | null;
    linear_regr_img_b64: string;
    linear_regr_params: string;
    linear_regr_summary: string;
  };
  status: number;
}

export interface Column {
  corr_to_price: number | null;
  price_data: number[];
  target_data: number[];
  corrs_to_shifted_prices: { label: string; corr: number }[];
  rows: number[];
  null_count: number;
  stats?: StatsCol;
  kline_open_time: number[];
}

export interface BacktestBalance {
  cash: number;
  kline_open_time: number;
  benchmark_price: number;
  portfolio_worth: number;
  position: number;
  prediction: number;
  short_debt: number;
  price: number;
}

export interface BacktestObject {
  id: number;
  dataset_name: string;
  open_trade_cond: string;
  close_trade_cond: string;
  use_time_based_close: boolean;
  use_profit_based_close: boolean;
  use_stop_loss_based_close: boolean;
  is_short_selling_strategy: boolean;
  is_long_short_strategy: boolean;
  klines_until_close: number;
  long_side_profit_factor: number;
  short_side_profit_factor: number;
  name: string;
  data: BacktestBalance[];
  trade_count: number;
  profit_factor: number;
  gross_profit: number;
  asset_universe_size: number;
  gross_loss: number;
  model_weights_id: number;
  train_job_id: number;
  dataset_id: number;
  start_balance: number;
  end_balance: number;
  result_perc: number;
  mean_return_perc: number;
  mean_hold_time_sec: number;
  take_profit_threshold_perc: number;
  stop_loss_threshold_perc: number;
  backtest_range_start: number;
  backtest_range_end: number;
  best_trade_result_perc: number;
  worst_trade_result_perc: number;
  buy_and_hold_result_net: number;
  buy_and_hold_result_perc: number;
  share_of_winning_trades_perc: number;
  share_of_losing_trades_perc: number;
  max_drawdown_perc: number;
  cagr: number;
  market_exposure_time: number;
  risk_adjusted_return: number;
  buy_and_hold_cagr: number;
  sharpe_ratio: number;
  probabilistic_sharpe_ratio: number;
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

export interface PairTrade {
  id: number;
  buy_trade_id: number;
  sell_trade_id: number;
  backtest_id: number;
  gross_result: number;
  percent_result: number;
  open_time: number;
  close_time: number;
  history: string;
}

export interface FetchBacktestByIdRes {
  data: BacktestObject;
  trades: Trade[];
  balance_history: BacktestBalance[];
  pair_trades: PairTrade[];
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

export interface CodePreset {
  id: number;
  code: string;
  label?: string;
  category: string;
  name: string;
  description?: string;
}

export interface DataTransformation {
  id: number;
  dataset_id: number;
  created_at: Date;
  updated_at: Date;
  transformation_code: string;
  name: string;
}

export interface MassBacktest {
  id: number;
  original_backtest_id: number;

  name: string;
  backtest_ids: number[];
}

export interface PortfolioHistoryTerse {
  kline_open_time: number;
  portfolio_worth: number;
}

export interface FetchBulkBacktests {
  data: BacktestObject[];
  equity_curves: { [key: number]: PortfolioHistoryTerse[] }[];
  id_to_dataset_name_map: { [key: number]: string };
}
