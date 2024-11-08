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
  data_transformation: {
    root: "/data-transformation",
  },
  stocks: {
    root: "/stocks",
    get_symbol_list: "/stocks/get-symbols",
  },
};

export const PRED_SERV_API = {
  v1_strategy: "/v1/strategy",
  v1_trade: "/v1/trade",
  v1_logs: "/v1/log",
  v1_account: "/v1/acc",
  v1_api_key: "/v1/api-key",
  v1_longshort: "/v1/longshort",
};

export let PRED_SERV_BASE_URL: string;
fetchEnvVar(ENV_VAR_KEYS.pred_server_uri).then((uri) => {
  PRED_SERV_BASE_URL = uri as string;
});

export const LOCAL_API_URI = CONSTANTS.base_url;

export const LOCAL_API_URL = {
  tables: LOCAL_API_URI + LOCAL_API.dataset.tables,
  get_table: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}`,
  get_column: (datasetName: string, columnName: string) =>
    LOCAL_API_URI +
    LOCAL_API.dataset.root +
    `/${datasetName}/col-info/${columnName}`,
  set_time_column: (datasetName: string) =>
    LOCAL_API_URI +
    LOCAL_API.dataset.root +
    `/${datasetName}/update-timeseries-col`,
  set_dataset_name: (datasetName: string) =>
    LOCAL_API_URI +
    LOCAL_API.dataset.root +
    `/${datasetName}/update-dataset-name`,
  binance_get_all_tickers: LOCAL_API_URI + LOCAL_API.binance.get_all_tickers,
  binance_fetch_klines: LOCAL_API_URI + LOCAL_API.binance.fetch_klines,
  ws_streams_log: LOCAL_API.streams.log,
  rename_column: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/rename-column`,
  add_columns: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/add-columns`,
  delete_dataset_cols: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/delete-cols`,
  exec_python_on_column: (datasetName: string, columnName: string) =>
    LOCAL_API_URI +
    LOCAL_API.dataset.root +
    `/${datasetName}/exec-python/${columnName}`,
  exec_python_on_dataset: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/exec-python`,
  create_model: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/models/create`,
  fetch_dataset_models: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/models`,
  fetch_model_by_id: (modelId: number) =>
    LOCAL_API_URI + LOCAL_API.model.root + `/${modelId}`,
  create_train_job: (modelName: string) =>
    LOCAL_API_URI + LOCAL_API.model.root + `/${modelName}/create-train`,
  fetch_all_training_metadata: (modelName: string) =>
    LOCAL_API_URI + LOCAL_API.model.root + `/${modelName}/trains`,
  stop_train: (trainJobId: string) =>
    LOCAL_API_URI + LOCAL_API.model.root + `/train/stop/${trainJobId}`,
  fetch_train_job_detailed: (trainJobId: string) =>
    LOCAL_API_URI + LOCAL_API.model.root + `/train/${trainJobId}/detailed`,
  fetch_backtests_by_dataset: (datasetId?: number) =>
    LOCAL_API_URI + LOCAL_API.backtest.root + `/dataset/${datasetId}`,
  fetch_backtest_by_id: (backtestId: number) =>
    LOCAL_API_URI + LOCAL_API.backtest.root + `/${backtestId}`,
  create_backtest: (trainJobId: string) =>
    LOCAL_API_URI + LOCAL_API.model.root + `/backtest/${trainJobId}/run`,
  setTargetColumn: (datasetName: string, targetColumn: string) =>
    LOCAL_API_URI +
    LOCAL_API.dataset.root +
    `/${datasetName}/target-column?target_column=${targetColumn}`,
  setPriceColumn: (datasetName: string, priceColumn: string) =>
    LOCAL_API_URI +
    LOCAL_API.dataset.root +
    `/${datasetName}/price-column?price_column=${priceColumn}`,
  createDatasetCopy: (datasetName: string, copyName: string) =>
    LOCAL_API_URI +
    LOCAL_API.dataset.root +
    `/${datasetName}/copy?new_dataset_name=${copyName}`,
  fetchTrainjobBacktests: (trainJobId: string) =>
    LOCAL_API_URI + LOCAL_API.model.root + `/backtest/${trainJobId}`,
  fetchDatasetPagination: (
    datasetName: string,
    page: number,
    pageSize: number
  ) => {
    return (
      LOCAL_API_URI +
      LOCAL_API.dataset.root +
      `/${datasetName}/pagination/${page}/${pageSize}`
    );
  },
  downloadDataset: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/download`,
  backtest: LOCAL_API_URI + LOCAL_API.backtest.root,
  createCodePreset: () => LOCAL_API_URI + LOCAL_API.code_preset.root,
  putCodePreset: () => LOCAL_API_URI + LOCAL_API.code_preset.root,
  deleteCodePreset: (id: number) =>
    LOCAL_API_URI + LOCAL_API.code_preset.root + `/${id}`,
  fetchCodePresets: () => LOCAL_API_URI + LOCAL_API.code_preset.root + "/all",
  deleteManyBacktest: (listOfIds: number[]) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    "/delete-many" +
    `?list_of_ids=${JSON.stringify(listOfIds)}`,
  massBacktest: () =>
    LOCAL_API_URI + LOCAL_API.backtest.root + "/mass-backtest",
  downloadBacktestSummary: (backtestId: number) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    `/${backtestId}` +
    "/detailed-summary",
  downloadBacktestSummaryMassSim: (backtestId: number) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    `/multistrategy/detailed-summary/${backtestId}`,
  massBacktestsByBacktestId: (backtestId: number) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    `/mass-backtest/by-backtest/${backtestId}`,
  massBacktestSymbols: (backtestId: number) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    `/mass-backtest/long-short/symbols/${backtestId}`,
  massBacktestTransformations: (backtestId: number) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    `/mass-backtest/long-short/transformations/${backtestId}`,
  massBacktestById: (massBacktestId: number) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    `/mass-backtest/${massBacktestId}`,

  fetchManyBacktestsById: (
    listOfBacktestIds: number[],
    includeEquityCurve: boolean
  ) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    `?list_of_ids=${JSON.stringify(
      listOfBacktestIds
    )}&include_equity_curve=${includeEquityCurve}`,
  downloadMassBacktestSummaryFile: (listOfBacktestIds: number[]) =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    "/mass-backtest/combined/summary?list_of_ids=" +
    `${JSON.stringify(listOfBacktestIds)}`,
  createRuleBasedMassBacktest: () =>
    LOCAL_API_URI + LOCAL_API.backtest.root + "/rule-based-on-universe",
  fetchDataTransformations: () =>
    LOCAL_API_URI + LOCAL_API.data_transformation.root,
  createDataTransformation: () =>
    LOCAL_API_URI + LOCAL_API.data_transformation.root,
  createMassPairTradeSim: () =>
    LOCAL_API_URI + LOCAL_API.backtest.root + "/long-short-backtest",
  fetchLongShortBacktests: () =>
    LOCAL_API_URI + LOCAL_API.backtest.root + "/mass-backtest/long-short/fetch",
  fetchRuleBasedMassBacktests: () =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    "/mass-backtest/rule-based/on-asset-universe",
  fetchMultiStrategyBacktests: () =>
    LOCAL_API_URI +
    LOCAL_API.backtest.root +
    "/mass-backtest/rule-based/multi-strat",
  fetchEpochValidationPreds: (trainJobId: number, epochNr: number) =>
    LOCAL_API_URI +
    LOCAL_API.model.root +
    `/train/${trainJobId}/epoch/${epochNr}`,
  createMlBasedBacktest: () =>
    LOCAL_API_URI + LOCAL_API.backtest.root + "/ml-based",
  fetchMlModelColumns: (modelId: number | undefined) =>
    LOCAL_API_URI + LOCAL_API.model.root + `/${modelId}/columns`,
  cloneIndicators: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/clone-indicators`,
  stocksSymbolList: () => LOCAL_API_URI + LOCAL_API.stocks.get_symbol_list,
  saveYfinanceKlines: (symbol: string) =>
    LOCAL_API_URI + LOCAL_API.stocks.root + "/yfinance/dataset/" + symbol,
  createMultiStrategyBacktest: () =>
    LOCAL_API_URI + LOCAL_API.backtest.root + "/rule-based-multi-strat",
  fetchDatasetOhlcvCols: (datasetName: string) =>
    LOCAL_API_URI + LOCAL_API.dataset.root + `/${datasetName}/ohlcv-columns`,
};

export const PRED_SERVER_URLS = {
  strategyEndpoint: () => PRED_SERV_BASE_URL + PRED_SERV_API.v1_strategy,
  createApiKeyEndpoint: () => PRED_SERV_BASE_URL + PRED_SERV_API.v1_api_key,
  deployPairtradeEndpoint: () =>
    PRED_SERV_BASE_URL + PRED_SERV_API.v1_longshort,
  deployStrategyGroupEndpoint: () =>
    PRED_SERV_BASE_URL + PRED_SERV_API.v1_strategy + "/strategy-group",
};
