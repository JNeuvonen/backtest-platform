import { UseQueryResult, useQuery } from "@tanstack/react-query";
import { QUERY_KEYS } from "../../utils/query-keys";
import {
  fetchAllTickers,
  fetchColumn,
  fetchDataset,
  fetchDatasetModels,
  fetchDatasets,
} from "../requests";
import {
  BinanceTickersResponse,
  ColumnResponse,
  DatasetModelResponse,
  DatasetResponse,
  DatasetsResponse,
} from "./response-types";

export function useDatasetsQuery(): UseQueryResult<DatasetsResponse, unknown> {
  return useQuery<DatasetsResponse, unknown>({
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
): UseQueryResult<DatasetResponse, unknown> {
  return useQuery<DatasetResponse, unknown>({
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
