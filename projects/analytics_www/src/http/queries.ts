import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { BalanceSnapshot } from "common_js";
import { QUERY_KEYS } from "src/utils";
import { fetchBalanceSnapshots } from "./requests";

export function useBalanceSnapshotsQuery(): UseQueryResult<
  BalanceSnapshot[],
  unknown
> {
  return useQuery<BalanceSnapshot[], unknown>({
    queryKey: [QUERY_KEYS.fetch_balance_snapshots],
    queryFn: fetchBalanceSnapshots,
  });
}
