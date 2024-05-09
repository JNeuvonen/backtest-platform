import { UseQueryResult, useQuery } from "@tanstack/react-query";
import { QUERY_KEYS } from "../../utils/query-keys";
import {
  AllTrainingMetadata,
  TrainDataDetailed,
  fetchAllTickers,
  fetchAllTrainingMetadataForModel,
  fetchBacktestById,
  fetchBacktestsByDataset,
  fetchCodePresets,
  fetchColumn,
  fetchDataTransformations,
  fetchDataset,
  fetchDatasetModels,
  fetchDatasets,
  fetchEpochValidationPreds,
  fetchLongShortBacktests,
  fetchManyBacktestsById,
  fetchMassBacktestById,
  fetchMassbacktestsById,
  fetchModelById,
  fetchModelTrainColumns,
  fetchTrainjobBacktests,
  fetchTrainjobDetailed,
} from "../requests";
import {
  BacktestObject,
  BinanceTickersResponse,
  CodePreset,
  ColumnResponse,
  DataTransformation,
  Dataset,
  DatasetMetadata,
  DatasetModel,
  EpochPredictionTick,
  FetchBacktestByIdRes,
  FetchBulkBacktests,
  MassBacktest,
} from "./response-types";
import { removeDuplicates } from "../../utils/number";

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
): UseQueryResult<DatasetModel[], unknown> {
  return useQuery<DatasetModel[], unknown>({
    queryKey: [QUERY_KEYS.fetch_datasets_models, datasetName],
    queryFn: () => fetchDatasetModels(datasetName),
  });
}

export function useModelQuery(
  modelId?: number
): UseQueryResult<DatasetModel | null, unknown> {
  return useQuery<DatasetModel | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_dataset_model_by_name, modelId],
    queryFn: () => fetchModelById(modelId as number),
    enabled: modelId !== undefined,
  });
}

export function useModelTrainMetadata(
  modelName: string
): UseQueryResult<AllTrainingMetadata | null, unknown> {
  return useQuery<AllTrainingMetadata | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_all_model_training_metadata, modelName],
    queryFn: () => fetchAllTrainingMetadataForModel(modelName),
    enabled: !!modelName,
  });
}

export function useTrainJobDetailed(
  trainJobId: string | number | undefined
): UseQueryResult<TrainDataDetailed | null, unknown> {
  return useQuery<TrainDataDetailed | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_train_job_detailed, trainJobId],
    queryFn: () => fetchTrainjobDetailed(trainJobId as string),
    enabled: trainJobId !== undefined,
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

export function useDatasetsBacktests(
  datasetId?: number
): UseQueryResult<BacktestObject[] | null, unknown> {
  return useQuery<BacktestObject[] | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_dataset_backtests],
    queryFn: () => fetchBacktestsByDataset(datasetId),
    enabled: datasetId !== undefined,
  });
}

export function useBacktestById(
  backtestId: number
): UseQueryResult<FetchBacktestByIdRes | null, unknown> {
  return useQuery<FetchBacktestByIdRes | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_backtest_by_id],
    queryFn: () => fetchBacktestById(backtestId),
    enabled: !!backtestId,
  });
}

export function useCodePresets(): UseQueryResult<CodePreset[] | null, unknown> {
  return useQuery<CodePreset[] | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_code_presets],
    queryFn: () => fetchCodePresets(),
  });
}

export function useMassbacktests(
  backtestId: number
): UseQueryResult<MassBacktest[] | null, unknown> {
  return useQuery<MassBacktest[] | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_mass_backtests_by_id, backtestId],
    queryFn: () => fetchMassbacktestsById(backtestId),
  });
}

export function useMassbacktest(
  massBacktestId: number
): UseQueryResult<MassBacktest | null, unknown> {
  return useQuery<MassBacktest | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_mass_backtest, massBacktestId],
    queryFn: () => fetchMassBacktestById(massBacktestId),
  });
}

export function useManyBacktests(
  listOfBacktestIds: number[],
  includeEquityCurve = false
): UseQueryResult<FetchBulkBacktests | null, unknown> {
  return useQuery<FetchBulkBacktests | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_many_backtests_by_id, listOfBacktestIds],
    queryFn: () =>
      fetchManyBacktestsById(
        removeDuplicates(listOfBacktestIds),
        includeEquityCurve
      ),
    enabled: listOfBacktestIds.length > 0,
  });
}

export function useDataTransformations(): UseQueryResult<
  DataTransformation[] | null,
  unknown
> {
  return useQuery<DataTransformation[] | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_many_backtests_by_id],
    queryFn: () => fetchDataTransformations(),
  });
}

export function useLongShortBacktests(): UseQueryResult<
  BacktestObject[] | null,
  unknown
> {
  return useQuery<BacktestObject[] | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_long_short_backtests],
    queryFn: () => fetchLongShortBacktests(),
  });
}

export function useEpochValPredictions(
  trainJobId: number | undefined,
  epochNumber: number | undefined
): UseQueryResult<EpochPredictionTick[] | null, unknown> {
  return useQuery<EpochPredictionTick[] | null, unknown>({
    queryKey: [
      QUERY_KEYS.fetch_epoch_validation_preds,
      trainJobId,
      epochNumber,
    ],
    queryFn: () => fetchEpochValidationPreds(trainJobId, epochNumber),
    enabled: trainJobId !== undefined && epochNumber !== undefined,
  });
}

export function useMLModelsColumns(
  modelId: number | undefined
): UseQueryResult<string[], unknown> {
  return useQuery<string[], unknown>({
    queryKey: [QUERY_KEYS.fetch_ml_models_cols, modelId],
    queryFn: () => fetchModelTrainColumns(modelId as number),
    enabled: modelId !== undefined,
  });
}
