import { CONSTANTS } from "../utils/constants";

const API = {
  tables: "/tables",
  dataset: "/dataset",
  binance: {
    get_all_tickers: "/binance/get-all-tickers",
    fetch_klines: "/binance/fetch-klines",
  },
  streams: {
    log: "ws://localhost:8000/streams/subscribe-log",
  },
};

const BASE_URL = CONSTANTS.base_url;

export const URLS = {
  get_tables: BASE_URL + API.tables,
  get_table: (datasetName: string) =>
    BASE_URL + API.dataset + `/${datasetName}`,
  get_column: (datasetName: string, columnName: string) =>
    BASE_URL + API.dataset + `/${datasetName}/col-info/${columnName}`,
  binance_get_all_tickers: BASE_URL + API.binance.get_all_tickers,
  binance_fetch_klines: BASE_URL + API.binance.fetch_klines,
  ws_streams_log: API.streams.log,
  rename_column: (datasetName: string) =>
    BASE_URL + API.dataset + `/${datasetName}/rename-column`,
};
