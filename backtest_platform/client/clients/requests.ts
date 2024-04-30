import { writeBinaryFile } from "@tauri-apps/api/fs";
import { AddColumnsReqPayload } from "../context/editor";
import { ModelDataPayload } from "../pages/data/model/create";
import { NullFillStrategy } from "../utils/constants";
import { LOCAL_API_URL, PRED_SERVER_URLS } from "./endpoints";
import { buildRequest } from "./fetch";
import { saveAs } from "file-saver";
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
import { save } from "@tauri-apps/api/dialog";
import { CodePresetBody } from "../components/SaveCodePresetPopover";
import { DeployStratForm } from "../pages/simulate/dataset/backtest/DeployStrategyForm";
import { predServerHeaders } from "./headers-utils";
import { CreateDataTransformationBody } from "../components/DataTransformationsControls";
import { BodyMassPairTradeSim } from "../context/masspairtrade/CreateNewDrawer";

export async function fetchDatasets() {
  const url = LOCAL_API_URL.tables;
  const res: DatasetsResponse = await buildRequest({ method: "GET", url });

  if (res?.res?.tables) {
    return res.res.tables;
  }
  return null;
}

export async function fetchDataset(datasetName: string) {
  const url = LOCAL_API_URL.get_table(datasetName);
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
  const url = LOCAL_API_URL.exec_python_on_column(datasetName, columnName);
  return buildRequest({ method: "POST", url, payload: { code } });
}

export async function execPythonOnDataset(
  datasetName: string,
  code: string,
  nullFillStrategy: NullFillStrategy
) {
  const url = LOCAL_API_URL.exec_python_on_dataset(datasetName);
  return buildRequest({
    method: "POST",
    url,
    payload: { code, null_fill_strategy: nullFillStrategy },
  });
}

export async function createModel(datasetName: string, body: ModelDataPayload) {
  const url = LOCAL_API_URL.create_model(datasetName);
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
  const url = LOCAL_API_URL.add_columns(datasetName);
  return buildRequest({ method: "POST", url, payload: payload });
}

export async function fetchColumn(datasetName: string, columnName: string) {
  const url = LOCAL_API_URL.get_column(datasetName, columnName);
  return buildRequest({ method: "GET", url });
}

export async function fetchAllTickers() {
  const url = LOCAL_API_URL.binance_get_all_tickers;
  return buildRequest({ method: "GET", url });
}

export async function fetchDatasetModels(datasetName: string) {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetch_dataset_models(datasetName),
  });

  return res;
}

export async function fetchModelByName(modelName: string) {
  const res: FetchModelByNameRes = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetch_model_by_name(modelName),
  });
  return res.res.model ? res.res.model : null;
}

export async function renameColumnName(
  datasetName: string,
  oldName: string,
  newName: string
) {
  const url = LOCAL_API_URL.rename_column(datasetName);
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
    url: LOCAL_API_URL.create_train_job(modelName),
    payload: trainJobForm,
  });
  return res;
}

export type AllTrainingMetadata = { train: TrainJob; weights: EpochInfo }[];

export async function fetchAllTrainingMetadataForModel(modelName: string) {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetch_all_training_metadata(modelName),
  });

  if (res.res) {
    return res.res["data"] as AllTrainingMetadata;
  }
  return null;
}

export async function stopTrain(trainJobId: string) {
  const res = await buildRequest({
    method: "POST",
    url: LOCAL_API_URL.stop_train(trainJobId),
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
    url: LOCAL_API_URL.fetch_train_job_detailed(trainJobId),
  });

  if (res.res) {
    return res.res["data"] as TrainDataDetailed;
  }
  return null;
}

export async function fetchTrainjobBacktests(trainJobId: string) {
  const res: BacktestsResponse = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetchTrainjobBacktests(trainJobId),
  });

  if (res.res) {
    return res.res["data"] as BacktestObject[];
  }
  return null;
}

export async function runBacktest(trainJobId: string, body: object) {
  const res = await buildRequest({
    method: "POST",
    url: LOCAL_API_URL.create_backtest(trainJobId),
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
    url: LOCAL_API_URL.setTargetColumn(datasetName, targetColumn),
  });
  return res;
}

export async function createCopyOfDataset(
  datasetName: string,
  copyName: string
) {
  const res = await buildRequest({
    method: "POST",
    url: LOCAL_API_URL.createDatasetCopy(datasetName, copyName),
  });
  return res;
}

export async function updatePriceColumnReq(
  datasetName: string,
  priceColumn: string
) {
  const res = await buildRequest({
    method: "PUT",
    url: LOCAL_API_URL.setPriceColumn(datasetName, priceColumn),
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
    url: LOCAL_API_URL.fetchDatasetPagination(datasetName, page, pageSize),
  });

  if (res.status === 200) {
    return res.res;
  }
  return res;
}

export async function saveDatasetFile(datasetName: string) {
  const response = await fetch(LOCAL_API_URL.downloadDataset(datasetName));
  if (!response.ok) throw new Error("Network response was not ok.");

  const blob = await response.blob();

  if (window.__TAURI__) {
    const arrayBuffer = await blob.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);
    const filePath = await save({ defaultPath: `${datasetName}.csv` });
    if (filePath) {
      await writeBinaryFile({ path: filePath, contents: uint8Array });
    }
  } else {
    saveAs(blob, `${datasetName}.csv`);
  }
}

export async function saveBacktestReport(
  backtestId: number,
  downloadName: string
) {
  const response = await fetch(
    LOCAL_API_URL.downloadBacktestSummary(backtestId)
  );
  if (!response.ok) throw new Error("Network response was not ok.");

  const blob = await response.blob();

  if (window.__TAURI__) {
    const arrayBuffer = await blob.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);
    const filePath = await save({ defaultPath: `${downloadName}.html` });
    if (filePath) {
      await writeBinaryFile({ path: filePath, contents: uint8Array });
    }
  } else {
    saveAs(blob, `${downloadName}.html`);
  }
}

export async function saveMassBacktestReport(backtestIds: number[]) {
  const response = await fetch(
    LOCAL_API_URL.downloadMassBacktestSummaryFile(backtestIds)
  );

  if (!response.ok) throw new Error("Network response was not ok.");

  const blob = await response.blob();

  if (window.__TAURI__) {
    const arrayBuffer = await blob.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);
    const filePath = await save({ defaultPath: "mass_backtest_summary.html" });
    if (filePath) {
      await writeBinaryFile({ path: filePath, contents: uint8Array });
    }
  } else {
    saveAs(blob, "bulk_backtest_summary.html");
  }
}

export async function removeDatasets(datasets: string[]) {
  const res = await buildRequest({
    method: "DELETE",
    url: LOCAL_API_URL.tables,
    payload: {
      dataset_names: datasets,
    },
  });

  return res;
}

interface CreateManualBacktest {
  name?: string;
  use_profit_based_close: boolean;
  use_stop_loss_based_close: boolean;
  is_short_selling_strategy: boolean;
  use_time_based_close: boolean;
  open_trade_cond: string;
  close_trade_cond: string;
  dataset_id: number;
  trading_fees_perc: number;
  slippage_perc: number;
  take_profit_threshold_perc: number;
  stop_loss_threshold_perc: number;
  short_fee_hourly: number;
  klines_until_close: null | number;
  backtest_data_range: number[];
}

export async function createManualBacktest(body: CreateManualBacktest) {
  const res = await buildRequest({
    method: "POST",
    url: LOCAL_API_URL.backtest,
    payload: body,
  });

  return res;
}

export async function fetchBacktestsByDataset(datasetId?: number) {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetch_backtests_by_dataset(datasetId),
  });

  if (res.status === 200) {
    return res.res["data"];
  }
  return res;
}

export async function fetchBacktestById(backtestId: number) {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetch_backtest_by_id(backtestId),
  });
  if (res.status === 200) {
    return res.res;
  }
  return res;
}

export async function createCodePreset(body: CodePresetBody) {
  const res = await buildRequest({
    method: "POST",
    url: LOCAL_API_URL.createCodePreset(),
    payload: body,
  });
  return res;
}

export async function fetchCodePresets() {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetchCodePresets(),
  });

  if (res.status === 200) {
    return res.res["data"];
  }
  return res;
}

export const setTargetColumn = async (
  targetCol: string,
  datasetName: string,
  successCallback?: () => void
) => {
  const res = await setTargetColumnReq(datasetName, targetCol);
  if (res.status === 200) {
    if (successCallback) {
      successCallback();
    }
  }
};

export const setBacktestPriceColumn = async (
  value: string,
  datasetName: string,
  successCallback?: () => void
) => {
  const res = await updatePriceColumnReq(datasetName, value);
  if (res.status === 200) {
    if (successCallback) {
      successCallback();
    }
  }
};

export const setKlineOpenTimeColumn = async (
  klineOpenTimeColumn: string,
  datasetName: string,
  successCallback?: () => void
) => {
  const url = LOCAL_API_URL.set_time_column(datasetName);
  const request = fetch(url, {
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      new_timeseries_col: klineOpenTimeColumn,
    }),
    method: "PUT",
  });
  request.then((res) => {
    if (res.status === 200) {
      if (successCallback) {
        successCallback();
      }
    }
  });
};

export const deleteManyBacktests = async (listOfBacktestIds: number[]) => {
  const res = await buildRequest({
    method: "DELETE",
    url: LOCAL_API_URL.deleteManyBacktest(listOfBacktestIds),
  });
  return res;
};

export const deployStrategyReq = async (
  apiKey: string,
  form: DeployStratForm
) => {
  const res = await buildRequest({
    method: "POST",
    url: PRED_SERVER_URLS.strategyEndpoint(),
    payload: form,
    options: {
      headers: predServerHeaders(apiKey),
    },
  });
  return res;
};

export const createPredServApiKey = async () => {
  const res = await buildRequest({
    method: "POST",
    url: PRED_SERVER_URLS.createApiKeyEndpoint(),
  });

  if (res.status === 200) {
    return res.res["data"]["key"];
  }
  return res;
};

interface PostMassBacktestBody {
  symbols: string[];
  original_backtest_id: number;
  fetch_latest_data: boolean;
}

export const postMassBacktest = async (body: PostMassBacktestBody) => {
  const res = await buildRequest({
    method: "POST",
    url: LOCAL_API_URL.massBacktest(),
    payload: body,
  });
  return res;
};

export const fetchMassbacktestsById = async (backtestId: number) => {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.massBacktestsByBacktestId(backtestId),
  });

  if (res.status === 200) {
    return res.res["data"];
  }
  return res;
};

export const fetchMassBacktestById = async (massBacktestId: number) => {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.massBacktestById(massBacktestId),
  });

  if (res.status === 200) {
    return res.res["data"];
  }
  return res;
};

export const fetchManyBacktestsById = async (
  listOfBacktestIds: number[],
  includeEquityCurve: boolean
) => {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetchManyBacktestsById(
      listOfBacktestIds,
      includeEquityCurve
    ),
  });

  if (res.status === 200) {
    return res.res;
  }
  return res;
};

export const fetchDataTransformations = async () => {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetchDataTransformations(),
  });

  if (res.status === 200) {
    return res.res["data"];
  }
  return res;
};

export const createDataTransformation = async (
  body: CreateDataTransformationBody
) => {
  const res = await buildRequest({
    method: "POST",
    url: LOCAL_API_URL.createDataTransformation(),
    payload: body,
  });
  return res;
};

export const createMassPairTradeSim = async (body: BodyMassPairTradeSim) => {
  const res = await buildRequest({
    method: "POST",
    url: LOCAL_API_URL.createMassPairTradeSim(),
    payload: body,
  });
  return res;
};

export const fetchLongShortBacktests = async () => {
  const res = await buildRequest({
    method: "GET",
    url: LOCAL_API_URL.fetchLongShortBacktests(),
  });
  if (res.status === 200) {
    return res.res["data"];
  }
  return res;
};
