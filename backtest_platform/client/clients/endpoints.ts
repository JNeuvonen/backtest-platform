import { CONSTANTS } from "../utils/constants";
import { fetchEnvVar } from "../utils/tauri";

const ENV_VAR_KEYS = {
  pred_server_uri: "REACT_APP_PRED_SERVER_URI",
};

const LOCAL_API = {
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
  backtest: {
    root: "/backtest",
  },
  code_preset: {
    root: "/code-preset",
  },
};

const PRED_SERV_API = {
  v1_strategy: "/v1/strategy",
  v1_trade: "/v1/trade",
  v1_logs: "/v1/log",
  v1_account: "/v1/acc",
};

let PRED_SERV_BASE_URL;
fetchEnvVar(ENV_VAR_KEYS.pred_server_uri).then((uri) => {
  PRED_SERV_BASE_URL = uri;
});

export const BASE_URL = CONSTANTS.base_url;

export const URLS = {
  tables: BASE_URL + LOCAL_API.dataset.tables,
  get_table: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}`,
  get_column: (datasetName: string, columnName: string) =>
    BASE_URL +
    LOCAL_API.dataset.root +
    `/${datasetName}/col-info/${columnName}`,
  set_time_column: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/update-timeseries-col`,
  set_dataset_name: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/update-dataset-name`,
  binance_get_all_tickers: BASE_URL + LOCAL_API.binance.get_all_tickers,
  binance_fetch_klines: BASE_URL + LOCAL_API.binance.fetch_klines,
  ws_streams_log: LOCAL_API.streams.log,
  rename_column: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/rename-column`,
  add_columns: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/add-columns`,
  delete_dataset_cols: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/delete-cols`,
  exec_python_on_column: (datasetName: string, columnName: string) =>
    BASE_URL +
    LOCAL_API.dataset.root +
    `/${datasetName}/exec-python/${columnName}`,
  exec_python_on_dataset: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/exec-python`,
  create_model: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/models/create`,
  fetch_dataset_models: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/models`,
  fetch_model_by_name: (modelName: string) =>
    BASE_URL + LOCAL_API.model.root + `/${modelName}`,
  create_train_job: (modelName: string) =>
    BASE_URL + LOCAL_API.model.root + `/${modelName}/create-train`,
  fetch_all_training_metadata: (modelName: string) =>
    BASE_URL + LOCAL_API.model.root + `/${modelName}/trains`,
  stop_train: (trainJobId: string) =>
    BASE_URL + LOCAL_API.model.root + `/train/stop/${trainJobId}`,
  fetch_train_job_detailed: (trainJobId: string) =>
    BASE_URL + LOCAL_API.model.root + `/train/${trainJobId}/detailed`,
  fetch_backtests_by_dataset: (datasetId?: number) =>
    BASE_URL + LOCAL_API.backtest.root + `/dataset/${datasetId}`,
  fetch_backtest_by_id: (backtestId: number) =>
    BASE_URL + LOCAL_API.backtest.root + `/${backtestId}`,
  create_backtest: (trainJobId: string) =>
    BASE_URL + LOCAL_API.model.root + `/backtest/${trainJobId}/run`,
  setTargetColumn: (datasetName: string, targetColumn: string) =>
    BASE_URL +
    LOCAL_API.dataset.root +
    `/${datasetName}/target-column?target_column=${targetColumn}`,
  setPriceColumn: (datasetName: string, priceColumn: string) =>
    BASE_URL +
    LOCAL_API.dataset.root +
    `/${datasetName}/price-column?price_column=${priceColumn}`,
  createDatasetCopy: (datasetName: string, copyName: string) =>
    BASE_URL +
    LOCAL_API.dataset.root +
    `/${datasetName}/copy?new_dataset_name=${copyName}`,
  fetchTrainjobBacktests: (trainJobId: string) =>
    BASE_URL + LOCAL_API.model.root + `/backtest/${trainJobId}`,

  fetchDatasetPagination: (
    datasetName: string,
    page: number,
    pageSize: number
  ) => {
    return (
      BASE_URL +
      LOCAL_API.dataset.root +
      `/${datasetName}/pagination/${page}/${pageSize}`
    );
  },
  downloadDataset: (datasetName: string) =>
    BASE_URL + LOCAL_API.dataset.root + `/${datasetName}/download`,
  backtest: BASE_URL + LOCAL_API.backtest.root,
  createCodePreset: () => BASE_URL + LOCAL_API.code_preset.root,
  fetchCodePresets: () => BASE_URL + LOCAL_API.code_preset.root + "/all",
  deleteManyBacktest: (listOfIds: number[]) =>
    BASE_URL +
    LOCAL_API.backtest.root +
    "/delete-many" +
    `?list_of_ids=${JSON.stringify(listOfIds)}`,
};
