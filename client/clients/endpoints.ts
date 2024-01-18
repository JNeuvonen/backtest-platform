import { CONSTANTS } from "../utils/constants";

const API = {
  tables: "/tables",
  dataset: {
    root: "/dataset",
    tables: "/dataset/tables",
  },
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
  get_tables: BASE_URL + API.dataset.tables,
  get_table: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}`,
  get_column: (datasetName: string, columnName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/col-info/${columnName}`,
  set_time_column: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/update-timeseries-col`,
  set_dataset_name: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/update-dataset-name`,
  binance_get_all_tickers: BASE_URL + API.binance.get_all_tickers,
  binance_fetch_klines: BASE_URL + API.binance.fetch_klines,
  ws_streams_log: API.streams.log,
  rename_column: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/rename-column`,
  add_columns: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/add-columns`,
  delete_dataset_cols: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/delete-cols`,

  execute_python_on_dataset: (datasetName: string, columnName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/exec-python/${columnName}`,
};
