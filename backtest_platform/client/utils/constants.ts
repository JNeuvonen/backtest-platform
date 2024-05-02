export const PATH_KEYS = {
  dataset: ":datasetName",
  column: ":columnName",
  model: ":modelName",
  train: ":trainJobId",
  backtest: ":backtestId",
  mass_backtest: ":massBacktestId",
  mass_pair_trade_backtest: ":massPairTradeBacktestId",
};

export const CODE = {
  INDENT: "    ",
  GET_DATASET_EXAMPLE: "dataset = get_dataset() #pandas dataframe",
  COL_SYMBOL: "column_name",
  DATASET_SYMBOL: "dataset",
};

export const COPILOT_LINKS = {
  trade_criteria:
    "https://chat.openai.com/g/g-K8Yblv0nE-trade-criteria-copilot",
};

export const PATHS = {
  data: {
    index: "/data",
    dataset: {
      index: `/data/datasets/${PATH_KEYS.dataset}/info`,
      column: `/data/datasets/${PATH_KEYS.dataset}/info/${PATH_KEYS.column}`,
      editor: `/data/datasets/${PATH_KEYS.dataset}/editor`,
    },
    model: {
      index: `/data/model/${PATH_KEYS.dataset}/`,
      info: `/data/model/${PATH_KEYS.dataset}/info/${PATH_KEYS.model}`,
      train: `/data/model/${PATH_KEYS.dataset}/info/${PATH_KEYS.model}/${PATH_KEYS.train}`,
    },
  },
  simulate: {
    path: "/simulate",
    select_dataset: "/simulate/select-dataset",
    dataset: `/simulate/${PATH_KEYS.dataset}`,
    backtest: `/simulate/${PATH_KEYS.dataset}/backtest/${PATH_KEYS.backtest}`,
    bulk_long_short: "/simulate/bulk-sim/long-short",
  },
  mass_backtest: {
    root: `/simulate/mass-backtests/${PATH_KEYS.backtest}`,
    pairtrade: `/simulate/mass-backtests/long-short/${PATH_KEYS.mass_pair_trade_backtest}`,
    backtest: `/simulate/mass-backtests/result/${PATH_KEYS.mass_backtest}`,
  },
  train: `/data/train-job/${PATH_KEYS.train}`,
  settings: "/settings",
};

export const LINKS = {
  datasets: "Data",
  simulate: "Simulate",
};

export const CONSTANTS = {
  base_url: "http://localhost:8000",
};

export const LAYOUT = {
  side_nav_width_px: 110,
  layout_padding_px: 16,
  inner_side_nav_width_px: 120,
  training_toolbar_height: 50,
};

export const TAURI_COMMANDS = {
  fetch_env: "fetch_env",
  fetch_platform: "fetch_platform",
};

export const BINANCE = {
  kline_options: {
    interval_1m: "1m",
    interval_3m: "3m",
    interval_5m: "5m",
    interval_15m: "15m",
    interval_30m: "30m",
    interval_1h: "1h",
    interval_2h: "2h",
    interval_4h: "4h",
    interval_6h: "6h",
    interval_8h: "8h",
    interval_12h: "12h",
    interval_1d: "1d",
    interval_3d: "3d",
    interval_1w: "1w",
    interval_1M: "1M",
  },
};

export const GET_KLINE_OPTIONS = () => {
  const klineOptions: string[] = Object.values(BINANCE.kline_options);
  return klineOptions;
};

export type BACKEND_MSG_SIGNALS =
  | "SIGNAL_OPEN_TRAINING_TOOLBAR"
  | "SIGNAL_CLOSE_TOOLBAR"
  | "SIGNAL_EPOCH_COMPLETE";

export const SIGNAL_OPEN_TRAINING_TOOLBAR: BACKEND_MSG_SIGNALS =
  "SIGNAL_OPEN_TRAINING_TOOLBAR";
export const SIGNAL_CLOSE_TOOLBAR: BACKEND_MSG_SIGNALS = "SIGNAL_CLOSE_TOOLBAR";
export const SIGNAL_EPOCH_COMPLETE = "SIGNAL_EPOCH_COMPLETE";

export const DOM_EVENT_CHANNELS = {
  refetch_all_datasets: "refetch_all_datasets",
  refetch_dataset: "refetch_dataset",
  refetch_dataset_columns: "refetch_dataset_columns",
  refetch_component: "refetch_component",
};

export type NullFillStrategy = "CLOSEST" | "MEAN" | "ZERO" | "NONE";
export type ScalingStrategy = "MIN-MAX" | "STANDARD" | "NONE";

export const NULL_FILL_STRATEGIES: {
  value: NullFillStrategy;
  label: string;
}[] = [
  { value: "CLOSEST", label: "Closest" },
  { value: "MEAN", label: "Mean" },
  { value: "ZERO", label: "Zero" },
  { value: "NONE", label: "None" },
];

export const SCALING_STRATEGIES: {
  value: ScalingStrategy;
  label: string;
}[] = [
  { value: "MIN-MAX", label: "Min-max scaling" },
  { value: "STANDARD", label: "Standard scaling" },
];

export const DOM_IDS = {
  select_null_fill_strat: "select-null-fill-strat",
};

export const CODE_PRESET_CATEGORY = {
  backtest_long_cond: "backtest_long_condition",
  backtest_close_long_ccond: "backtest_close_long_condition",
  backtest_pair_trade_exit_cond: "backtest_pair_trade_exit_cond",
  backtest_create_columns: "backtest_create_columns",
  strategy_deploy_enter_trade: "strategy_enter_trade_code",
  strategy_deploy_exit_trade: "strategy_enter_trade_code",
  strategy_deploy_fetch_datasources: "strategy_fetch_datasources",
  strategy_deploy_data_transformations: "stategy_data_transformations",
};

export class CodeHelper {
  indentLevel: number;
  code: string;

  constructor() {
    this.indentLevel = 0;
    this.code = "";
  }

  appendLine(line: string) {
    let newLine = "";
    for (let i = 0; i < this.indentLevel; i++) {
      newLine += CODE.INDENT;
    }
    newLine += line + "\n";
    this.code += newLine;
  }

  addIndent() {
    this.indentLevel += 1;
  }
  resetIndent() {
    this.indentLevel = 0;
  }
  reduceIndent() {
    this.indentLevel -= 1;
  }

  get() {
    return this.code.replace(/\n$/, "");
  }
}

export const formatValidationSplit = (valSplit: string): [number, number] => {
  const parts = valSplit.split(",");
  return [Number(parts[0]), Number(parts[1])];
};
