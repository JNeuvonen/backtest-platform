export const PATHS = {
  datasets: {
    path: "/datasets",
    subpaths: {
      available: {
        path: "/datasets/all",
      },
      binance: {
        path: "/datasets/binance",
      },
      stock_market: {
        path: "/datasets/stock-market",
      },
    },
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
  const klineOptions: string[] = [];
  for (const [_, value] of Object.entries(BINANCE.kline_options)) {
    klineOptions.push(value);
  }
  return klineOptions;
};

export const DOM_EVENT_CHANNELS = {
  refetch_all_datasets: "refetch_all_datasets",
  refetch_dataset_columns: "refetch_dataset_columns",
};
