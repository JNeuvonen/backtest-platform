import { useQuery } from "@tanstack/react-query";
import { FETCH_DATASETS } from "../utils/query-keys";

export const fetchDatasets = () => {
  return useQuery([FETCH_DATASETS], fetchPosts);
};
