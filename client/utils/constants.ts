export const PATHS = {
  datasets: {
    path: "/datasets",
    subpaths: {
      available: {
        path: "all",
      },
      binance: {
        path: "binance",
      },
      stock_market: {
        path: "stock-market",
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
