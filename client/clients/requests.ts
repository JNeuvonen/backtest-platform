import { AddColumnsReqPayload } from "../context/editor";
import { ModelDataPayload } from "../pages/data/model/create";
import { NullFillStrategy } from "../utils/constants";
import { URLS } from "./endpoints";
import { buildRequest } from "./fetch";
import {
  BacktestObject,
  BacktestsResponse,
  DatasetModel,
  DatasetResponse,
  DatasetUtils,
  DatasetsResponse,
  EpochInfo,
  FetchModelByNameRes,
  TrainJob,
} from "./queries/response-types";

export async function fetchDatasets() {
  const url = URLS.get_tables;
  const res: DatasetsResponse = await buildRequest({ method: "GET", url });

  if (res?.res?.tables) {
    return res.res.tables;
  }
  return null;
}

export async function fetchDataset(datasetName: string) {
  const url = URLS.get_table(datasetName);
  const res: DatasetResponse = await buildRequest({ method: "GET", url });

  if (res?.res?.dataset) {
    return res.res.dataset;
  }

  return null;
}

export async function execPythonOnDatasetCol(
  datasetName: string,
  columnName: string,
  code: string
) {
  const url = URLS.exec_python_on_column(datasetName, columnName);
  return buildRequest({ method: "POST", url, payload: { code } });
}

export async function execPythonOnDataset(
  datasetName: string,
  code: string,
  nullFillStrategy: NullFillStrategy
) {
  const url = URLS.exec_python_on_dataset(datasetName);
  return buildRequest({
    method: "POST",
    url,
    payload: { code, null_fill_strategy: nullFillStrategy },
  });
}

export async function createModel(datasetName: string, body: ModelDataPayload) {
  const url = URLS.create_model(datasetName);
  return buildRequest({
    method: "POST",
    url,
    payload: body,
  });
}

export async function addColumnsToDataset(
  datasetName: string,
  payload: AddColumnsReqPayload
) {
  const url = URLS.add_columns(datasetName);
  return buildRequest({ method: "POST", url, payload: payload });
}

export async function fetchColumn(datasetName: string, columnName: string) {
  const url = URLS.get_column(datasetName, columnName);
  return buildRequest({ method: "GET", url });
}

export async function fetchAllTickers() {
  const url = URLS.binance_get_all_tickers;
  return buildRequest({ method: "GET", url });
}

export async function fetchDatasetModels(datasetName: string) {
  const res = await buildRequest({
    method: "GET",
    url: URLS.fetch_dataset_models(datasetName),
  });

  return res;
}

export async function fetchModelByName(modelName: string) {
  const res: FetchModelByNameRes = await buildRequest({
    method: "GET",
    url: URLS.fetch_model_by_name(modelName),
  });
  return res.res.model ? res.res.model : null;
}

export async function renameColumnName(
  datasetName: string,
  oldName: string,
  newName: string
) {
  const url = URLS.rename_column(datasetName);
  return buildRequest({
    method: "POST",
    url,
    payload: {
      old_col_name: oldName,
      new_col_name: newName,
    },
  });
}

export async function createTrainJob(modelName: string, trainJobForm: object) {
  const res = await buildRequest({
    method: "POST",
    url: URLS.create_train_job(modelName),
    payload: trainJobForm,
  });
  return res;
}

export type AllTrainingMetadata = { train: TrainJob; weights: EpochInfo }[];

export async function fetchAllTrainingMetadataForModel(modelName: string) {
  const res = await buildRequest({
    method: "GET",
    url: URLS.fetch_all_training_metadata(modelName),
  });

  if (res.res) {
    return res.res["data"] as AllTrainingMetadata;
  }
  return null;
}

export async function stopTrain(trainJobId: string) {
  const res = await buildRequest({
    method: "POST",
    url: URLS.stop_train(trainJobId),
  });
  return res;
}

export type TrainDataDetailed = {
  dataset_metadata: DatasetUtils;
  dataset_columns: string[];
  model: DatasetModel;
  train_job: TrainJob;
  epochs: EpochInfo[];
};

export async function fetchTrainjobDetailed(trainJobId: string) {
  const res = await buildRequest({
    method: "GET",
    url: URLS.fetch_train_job_detailed(trainJobId),
  });

  if (res.res) {
    return res.res["data"] as TrainDataDetailed;
  }
  return null;
}

export async function fetchTrainjobBacktests(trainJobId: string) {
  const res: BacktestsResponse = await buildRequest({
    method: "GET",
    url: URLS.fetchTrainjobBacktests(trainJobId),
  });

  if (res.res) {
    return res.res["data"] as BacktestObject[];
  }
  return null;
}

export async function runBacktest(trainJobId: string, body: object) {
  const res = await buildRequest({
    method: "POST",
    url: URLS.create_backtest(trainJobId),
    payload: body,
  });
  return res;
}

export async function setTargetColumnReq(
  datasetName: string,
  targetColumn: string
) {
  const res = await buildRequest({
    method: "PUT",
    url: URLS.setTargetColumn(datasetName, targetColumn),
  });
  return res;
}

export async function createCopyOfDataset(
  datasetName: string,
  copyName: string
) {
  const res = await buildRequest({
    method: "POST",
    url: URLS.createDatasetCopy(datasetName, copyName),
  });
  return res;
}

export async function updatePriceColumnReq(
  datasetName: string,
  priceColumn: string
) {
  const res = await buildRequest({
    method: "PUT",
    url: URLS.setPriceColumn(datasetName, priceColumn),
  });
  return res;
}

export async function fetchDatasetPagination(
  datasetName: string,
  page: number,
  pageSize: number
) {
  const res = await buildRequest({
    method: "GET",
    url: URLS.fetchDatasetPagination(datasetName, page, pageSize),
  });

  if (res.status === 200) {
    return res.res;
  }
  return res;
}
