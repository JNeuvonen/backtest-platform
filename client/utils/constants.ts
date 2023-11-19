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

export const ENV = {
  base_url: "REACT_APP_BACKEND_URL",
};

export const TAURI_COMMANDS = {
  fetch_env: "fetch_env",
};
