import { UseQueryResult, useQuery } from "@tanstack/react-query";
import { FETCH_DATASETS } from "../utils/query-keys";
import { fetchDatasets } from "./requests";

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

export function useDatasetsQuery(): UseQueryResult<DatasetsResponse, unknown> {
  return useQuery<DatasetsResponse, unknown>({
    queryKey: [FETCH_DATASETS],
    queryFn: fetchDatasets,
  });
}
