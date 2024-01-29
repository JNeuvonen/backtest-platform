import { UseQueryResult, useQuery } from "@tanstack/react-query";
import { QUERY_KEYS } from "../../utils/query-keys";
import {
  AllTrainingMetadata,
  TrainDataDetailed,
  fetchAllTickers,
  fetchAllTrainingMetadataForModel,
  fetchColumn,
  fetchDataset,
  fetchDatasetModels,
  fetchDatasets,
  fetchModelByName,
  fetchTrainjobBacktests,
  fetchTrainjobDetailed,
} from "../requests";
import {
  BacktestObject,
  BinanceTickersResponse,
  ColumnResponse,
  Dataset,
  DatasetMetadata,
  DatasetModel,
  DatasetModelResponse,
} from "./response-types";

export function useDatasetsQuery(): UseQueryResult<
  DatasetMetadata[] | null,
  unknown
> {
  return useQuery<DatasetMetadata[] | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_datasets],
    queryFn: fetchDatasets,
  });
}

export function useBinanceTickersQuery(): UseQueryResult<
  BinanceTickersResponse,
  unknown
> {
  return useQuery<BinanceTickersResponse, unknown>({
    queryKey: [QUERY_KEYS.fetch_binance_tickers],
    queryFn: fetchAllTickers,
  });
}

export function useDatasetQuery(
  datasetName: string
): UseQueryResult<Dataset | null, unknown> {
  return useQuery<Dataset | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_datasets, datasetName],
    queryFn: () => fetchDataset(datasetName),
  });
}

export function useColumnQuery(
  datasetName: string,
  columnName: string
): UseQueryResult<ColumnResponse, unknown> {
  return useQuery<ColumnResponse, unknown>({
    queryKey: [QUERY_KEYS.fetch_column, datasetName, columnName],
    queryFn: () => fetchColumn(datasetName, columnName),
  });
}

export function useDatasetModelsQuery(
  datasetName: string
): UseQueryResult<DatasetModelResponse, unknown> {
  return useQuery<DatasetModelResponse, unknown>({
    queryKey: [QUERY_KEYS.fetch_datasets_models, datasetName],
    queryFn: () => fetchDatasetModels(datasetName),
  });
}

export function useModelQuery(
  modelName: string
): UseQueryResult<DatasetModel | null, unknown> {
  return useQuery<DatasetModel | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_dataset_model_by_name, modelName],
    queryFn: () => fetchModelByName(modelName),
  });
}

export function useModelTrainMetadata(
  modelName: string
): UseQueryResult<AllTrainingMetadata | null, unknown> {
  return useQuery<AllTrainingMetadata | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_all_model_training_metadata, modelName],
    queryFn: () => fetchAllTrainingMetadataForModel(modelName),
  });
}

export function useTrainJobDetailed(
  trainJobId: string
): UseQueryResult<TrainDataDetailed | null, unknown> {
  return useQuery<TrainDataDetailed | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_train_job_detailed, trainJobId],
    queryFn: () => fetchTrainjobDetailed(trainJobId),
  });
}

export function useTrainJobBacktests(
  trainJobId: string
): UseQueryResult<BacktestObject[] | null, unknown> {
  return useQuery<BacktestObject[] | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_trainjob_backtests, trainJobId],
    queryFn: () => fetchTrainjobBacktests(trainJobId),
  });
}
