import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { QUERY_KEYS } from "src/utils";
import { fetchBalanceSnapshots } from "./requests";

export function useBalanceSnapshotsQuery(): UseQueryResult<any, unknown> {
  return useQuery<any[] | null, unknown>({
    queryKey: [QUERY_KEYS.fetch_balance_snapshots],
    queryFn: fetchBalanceSnapshots,
  });
}
