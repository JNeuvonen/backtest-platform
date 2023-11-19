import { useQuery } from "@tanstack/react-query";
import { FETCH_DATASETS } from "../utils/query-keys";
import { fetchDatasets } from "./requests";

export function useDatasetsQuery() {
  return useQuery({
    queryKey: [FETCH_DATASETS],
    queryFn: fetchDatasets,
  });
}
