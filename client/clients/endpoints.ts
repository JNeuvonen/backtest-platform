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
  model: {
    root: "/model",
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
  exec_python_on_column: (datasetName: string, columnName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/exec-python/${columnName}`,
  exec_python_on_dataset: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/exec-python`,
  create_model: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/models/create`,
  fetch_dataset_models: (datasetName: string) =>
    BASE_URL + API.dataset.root + `/${datasetName}/models`,
  fetch_model_by_name: (modelName: string) =>
    BASE_URL + API.model.root + `/${modelName}`,
  create_train_job: (modelName: string) =>
    BASE_URL + API.model.root + `/${modelName}/create-train`,
  fetch_all_training_metadata: (modelName: string) =>
    BASE_URL + API.model.root + `/${modelName}/trains`,
  stop_train: (trainJobId: string) =>
    BASE_URL + API.model.root + `/train/stop/${trainJobId}`,
  fetch_train_job_detailed: (trainJobId: string) =>
    BASE_URL + API.model.root + `/train/${trainJobId}/detailed`,
  create_backtest: (trainJobId: string) =>
    BASE_URL + API.model.root + `/backtest/${trainJobId}/run`,
  setTargetColumn: (datasetName: string, targetColumn: string) =>
    BASE_URL +
    API.dataset.root +
    `/${datasetName}/target-column?target_column=${targetColumn}`,
  setPriceColumn: (datasetName: string, priceColumn: string) =>
    BASE_URL +
    API.dataset.root +
    `/${datasetName}/price-column?price_column=${priceColumn}`,
  createDatasetCopy: (datasetName: string, copyName: string) =>
    BASE_URL +
    API.dataset.root +
    `/${datasetName}/copy?new_dataset_name=${copyName}`,
  fetchTrainjobBacktests: (trainJobId: string) =>
    BASE_URL + API.model.root + `/backtest/${trainJobId}`,
};
