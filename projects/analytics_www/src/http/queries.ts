import { useQuery, UseQueryResult } from "@tanstack/react-query";
import {
  BalanceSnapshot,
  BinanceSymbolPrice,
  StrategiesResponse,
  StrategyGroupResponse,
} from "common_js";
import { QUERY_KEYS } from "src/utils";
import {
  fetchBalanceSnapshots,
  fetchBinancePriceInfo,
  fetchStrategies,
  fetchStrategyGroup,
} from "./requests";

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

export function useStrategyGroupQuery(
  groupName: string,
): UseQueryResult<StrategyGroupResponse, unknown> {
  return useQuery<StrategyGroupResponse, unknown>({
    queryKey: [QUERY_KEYS.fetch_balance_snapshots, groupName],
    queryFn: () => fetchStrategyGroup(groupName.toUpperCase()) as any,
  });
}

export function useBinanceSpotPriceInfo(): UseQueryResult<
  BinanceSymbolPrice[],
  unknown
> {
  return useQuery<BinanceSymbolPrice[], unknown>({
    queryKey: [QUERY_KEYS.fetch_binance_spot_price_info],
    queryFn: () => fetchBinancePriceInfo(),
    refetchInterval: 5000,
  });
}
