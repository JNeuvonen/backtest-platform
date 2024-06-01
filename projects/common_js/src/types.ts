export interface BalanceSnapshot {
  id: number;
  created_at: string;
  value: number;
  debt: number;
  btc_price: number;
  long_assets_value: number;
  margin_level: number;
  num_long_positions: number;
  num_short_positions: number;
  num_ls_positions: number;
}

export interface Strategy {
  id: number;
  active_trade_id?: number;
  strategy_group_id?: number;
  name: string;
  strategy_group?: string;
  created_at: Date;
  updated_at: Date;
  symbol: string;
  base_asset: string;
  quote_asset: string;
  enter_trade_code: string;
  exit_trade_code: string;
  fetch_datasources_code: string;
  candle_interval?: string;
  trade_quantity_precision: number;
  priority: number;
  num_req_klines: number;
  kline_size_ms?: number;
  last_kline_open_time_sec?: number;
  minimum_time_between_trades_ms?: number;
  maximum_klines_hold_time?: number;
  time_on_trade_open_ms: number;
  price_on_trade_open?: number;
  quantity_on_trade_open: number;
  remaining_position_on_trade: number;
  allocated_size_perc?: number;
  take_profit_threshold_perc?: number;
  stop_loss_threshold_perc?: number;
  use_time_based_close: boolean;
  use_profit_based_close: boolean;
  use_stop_loss_based_close: boolean;
  use_taker_order?: boolean;
  stop_processing_new_candles?: boolean;
  should_enter_trade: boolean;
  should_close_trade: boolean;
  should_calc_stops_on_pred_serv: boolean;
  is_on_pred_serv_err: boolean;
  is_paper_trade_mode: boolean;
  is_leverage_allowed: boolean;
  is_short_selling_strategy: boolean;
  is_disabled: boolean;
  is_in_close_only: boolean;
  is_in_position: boolean;
}

export interface LongShortGroup {
  id: number;
  name: string;
  candle_interval?: string;
  buy_cond?: string;
  sell_cond?: string;
  exit_cond?: string;
  num_req_klines?: number;
  max_simultaneous_positions?: number;
  klines_until_close?: number;
  kline_size_ms?: number;
  loan_retry_wait_time_ms?: number;
  max_leverage_ratio?: number;
  take_profit_threshold_perc?: number;
  stop_loss_threshold_perc?: number;
  is_disabled: boolean;
  use_time_based_close?: boolean;
  use_profit_based_close?: boolean;
  use_stop_loss_based_close?: boolean;
  use_taker_order?: boolean;
}

export interface Trade {
  id: number;
  symbol: string;
  strategy_id?: number;
  pair_trade_group_id: number;
  strategy_group_id: number;
  created_at: Date;
  updated_at: Date;
  open_time_ms: number;
  close_time_ms?: number;
  open_price: number;
  quantity: number;
  close_price?: number;
  net_result?: number;
  percent_result?: number;
  direction: string;
  info_ticks: TradeInfoTick[];
}

export interface TradeInfoTick {
  id: number;
  trade_id: number;
  price: number;
  kline_open_time_ms: number;
}

export interface StrategyGroup {
  id: number;
  name: string;

  created_at: Date;
  updated_at: Date;

  transformation_ids: string;
  is_disabled: boolean;
  is_close_only: boolean;
}

export interface LongShortTicker {
  id: number;
  long_short_group_id: number;

  symbol: string;
  base_asset: string;
  quote_asset: string;
  dataset_name: string;

  last_kline_open_time_sec?: number | null;
  trade_quantity_precision: number;

  is_valid_buy: boolean;
  is_valid_sell: boolean;
  is_on_pred_serv_err: boolean;
}

export interface LongShortPair {
  id: number;
  long_short_group_id: number;
  buy_ticker_id: number;
  sell_ticker_id: number;
  buy_side_trade_id: number;
  sell_side_trade_id: number;

  buy_ticker_dataset_name: string;
  sell_ticker_dataset_name: string;

  buy_symbol: string;
  sell_symbol: string;
  buy_base_asset: string;
  sell_base_asset: string;
  buy_quote_asset: string;
  sell_quote_asset: string;

  buy_qty_precision: number;
  sell_qty_precision: number;
  buy_open_time_ms: number;
  sell_open_time_ms: number;
  last_loan_attempt_fail_time_ms: number;

  buy_open_price: number;
  sell_open_price: number;
  buy_open_qty_in_base: number;
  buy_open_qty_in_quote: number;
  sell_open_qty_in_quote: number;
  debt_open_qty_in_base: number;

  is_no_loan_available_err: boolean;
  error_in_entering: boolean;
  in_position: boolean;
  should_close: boolean;
  is_trade_finished: boolean;
}

export interface StrategiesResponse {
  ls_strategies: LongShortGroup[];
  ls_pairs: LongShortPair[];
  ls_tickers: LongShortTicker[];
  directional_strategies: Strategy[];
  trades: Trade[];
  strategy_groups: StrategyGroup[];
}

export interface StrategyGroupResponse {
  strategy_group: StrategyGroup;
  strategies: Strategy[];
  trades: Trade[];
}

export interface BinanceSymbolPrice {
  symbol: string;
  price: string;
}
