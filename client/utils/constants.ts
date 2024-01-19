export const PATH_KEYS = {
  dataset: ":datasetName",
  column: ":columnName",
};

export const CODE = {
  INDENT: "    ",
  GET_DATASET_EXAMPLE: "dataset = get_dataset() #pandas dataframe",
  COL_SYMBOL: "column_name",
  DATASET_SYMBOL: "dataset",
};

export const PATHS = {
  datasets: {
    path: "/datasets",
    dataset: `/datasets/${PATH_KEYS.dataset}`,
    info: `/datasets/${PATH_KEYS.dataset}/info`,
    column: `/datasets/${PATH_KEYS.dataset}/info/${PATH_KEYS.column}`,
    editor: `/datasets/${PATH_KEYS.dataset}/editor`,
  },
  simulate: {
    path: "/simulate",
  },
};

export const LINKS = {
  datasets: "Datasets",
  simulate: "Simulate",
};

export const CONSTANTS = {
  base_url: "http://localhost:8000",
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

export const DOM_EVENT_CHANNELS = {
  refetch_all_datasets: "refetch_all_datasets",
  refetch_dataset: "refetch_dataset",
  refetch_dataset_columns: "refetch_dataset_columns",
};

export type NullFillStrategy = "CLOSEST" | "MEAN" | "ZERO" | "NONE";

export const NULL_FILL_STRATEGIES: {
  value: NullFillStrategy;
  label: string;
}[] = [
  { value: "CLOSEST", label: "Closest" },
  { value: "MEAN", label: "Mean" },
  { value: "ZERO", label: "Zero" },
  { value: "NONE", label: "None" },
];
