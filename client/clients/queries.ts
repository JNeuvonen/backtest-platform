import { UseQueryResult, useQuery } from "@tanstack/react-query";
import { QUERY_KEYS } from "../utils/query-keys";
import { fetchAllTickers, fetchDatasets } from "./requests";

export interface DatasetMetadata {
  columns: string[];
  start_date: number;
  end_date: number;
  table_name: string;
}

interface DatasetsResponse {
  res: {
    tables: DatasetMetadata[];
  };
  status: number;
}

export interface BinanceBasicTicker {
  symbol: string;
  price: number;
}

interface BinanceTickersResponse {
  res: {
    pairs: BinanceBasicTicker[];
  };
  status: number;
}

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
