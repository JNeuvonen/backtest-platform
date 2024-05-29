import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { BalanceSnapshot, StrategiesResponse } from "common_js";
import { QUERY_KEYS } from "src/utils";
import { fetchBalanceSnapshots, fetchStrategies } from "./requests";

export function useBalanceSnapshotsQuery(): UseQueryResult<
  BalanceSnapshot[],
  unknown
> {
  return useQuery<BalanceSnapshot[], unknown>({
    queryKey: [QUERY_KEYS.fetch_balance_snapshots],
    queryFn: fetchBalanceSnapshots,
  });
}

export function useStrategiesQuery(): UseQueryResult<
  StrategiesResponse,
  unknown
> {
  return useQuery<StrategiesResponse, unknown>({
    queryKey: [QUERY_KEYS.fetch_balance_snapshots],
    queryFn: fetchStrategies,
  });
}
