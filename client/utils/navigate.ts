import { NullFillStrategy, PATHS, PATH_KEYS } from "./constants";

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

export const getModelInfoPath = (datasetName: string, modelName: string) => {
  return PATHS.data.model.info
    .replace(PATH_KEYS.dataset, datasetName)
    .replace(PATH_KEYS.model, modelName);
};

export const getTrainJobPath = (
  datasetName: string,
  modelName: string,
  trainJobId: string
) => {
  return PATHS.data.model.train
    .replace(PATH_KEYS.dataset, datasetName)
    .replace(PATH_KEYS.model, modelName)
    .replace(PATH_KEYS.train, trainJobId);
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
