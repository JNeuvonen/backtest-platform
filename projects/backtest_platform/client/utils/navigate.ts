import {
  NullFillStrategy,
  PATHS,
  PATH_KEYS,
  ScalingStrategy,
} from "./constants";

export const getDatasetEditorPath = (datasetName: string) => {
  return (
    PATHS.data.dataset.editor.replace(PATH_KEYS.dataset, datasetName) +
    getQueryParamDefaultTab(1)
  );
};

export const getDatasetColumnInfoPath = (
  datasetName: string,
  columnName: string
) => {
  return PATHS.data.dataset.column
    .replace(PATH_KEYS.dataset, datasetName)
    .replace(PATH_KEYS.column, columnName);
};

export const getDatasetInfoPagePath = (datasetName: string) => {
  return PATHS.data.dataset.index.replace(PATH_KEYS.dataset, datasetName);
};

export const getModelInfoPath = (datasetName: string, modelId: number) => {
  return PATHS.data.model.info
    .replace(PATH_KEYS.dataset, datasetName)
    .replace(PATH_KEYS.model, String(modelId));
};

export const getMassbacktestTablesPath = (backtestId: string) => {
  return PATHS.mass_backtest.root.replace(PATH_KEYS.backtest, backtestId);
};

export const getInvidualMassBacktestPath = (massBacktestId: string) => {
  return PATHS.mass_backtest.backtest.replace(
    PATH_KEYS.mass_backtest,
    massBacktestId
  );
};

export const getTrainJobPath = (
  datasetName: string,
  modelId: number,
  trainJobId: string
) => {
  return PATHS.data.model.train
    .replace(PATH_KEYS.dataset, datasetName)
    .replace(PATH_KEYS.model, String(modelId))
    .replace(PATH_KEYS.train, trainJobId);
};

export const getDatasetBacktestPath = (dataset: string, backtestId: number) => {
  return PATHS.simulate.backtest
    .replace(PATH_KEYS.dataset, dataset)
    .replace(PATH_KEYS.backtest, String(backtestId));
};

export const getRuleBasedBacktestByIdPath = (backtestId: number) => {
  return PATHS.rule_based_on_universe.by_id.replace(
    PATH_KEYS.backtest,
    String(backtestId)
  );
};

export const getMlBasedBacktestPath = (dataset: string, backtestId: number) => {
  return PATHS.ml_based.backtest
    .replace(PATH_KEYS.dataset, dataset)
    .replace(PATH_KEYS.backtest, String(backtestId));
};

export const getPairTradeBacktestPath = (backtestId: number) => {
  return PATHS.mass_backtest.pairtrade.replace(
    PATH_KEYS.mass_pair_trade_backtest,
    String(backtestId)
  );
};

export const getTrainJobFromToolbar = (trainJobId: string) => {
  return PATHS.train.replace(PATH_KEYS.train, trainJobId);
};

export const getQueryParamDefaultTab = (idx: number) => {
  return `?defaultTab=${idx}`;
};

export const replaceNthPathItem = (nthItemFromEnd: number, newItem: string) => {
  const pathParts = window.location.pathname.split("/");
  pathParts[pathParts.length - 1 - nthItemFromEnd] = newItem;
  return pathParts.join("/");
};

export const nullFillStratToInt = (nullFillStrat: NullFillStrategy) => {
  switch (nullFillStrat) {
    case "NONE":
      return 1;

    case "ZERO":
      return 2;

    case "MEAN":
      return 3;

    case "CLOSEST":
      return 4;

    default:
      return 1;
  }
};

export const scalingStrategyToInt = (scalingStrategy: ScalingStrategy) => {
  switch (scalingStrategy) {
    case "NONE":
      return 1;

    case "MIN-MAX":
      return 2;

    case "STANDARD":
      return 3;

    default:
      return 1;
  }
};
