import { useQuery, UseQueryResult } from "@tanstack/react-query";
import {
  BalanceSnapshot,
  BinanceSymbolPrice,
  BinanceTickerPriceChange,
  BinanceUserAsset,
  LongShortGroupResponse,
  StrategiesResponse,
  StrategyGroupResponse,
  Trade,
} from "src/common_js";
import { QUERY_KEYS } from "src/utils";
import {
  fetchBalanceSnapshots,
  fetchBalanceSnapshots1DInterval,
  fetchBinance24hPriceChanges,
  fetchBinancePriceInfo,
  fetchLatestBalanceSnapshot,
  fetchLongShortGroup,
  fetchStrategies,
  fetchStrategyGroup,
  fetchUncompletedTrades,
  fetchUserAssets,
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
    queryKey: [QUERY_KEYS.fetch_strategies],
    queryFn: fetchStrategies,
  });
}

export function useStrategyGroupQuery(
  groupName: string,
): UseQueryResult<StrategyGroupResponse, unknown> {
  return useQuery<StrategyGroupResponse, unknown>({
    queryKey: [QUERY_KEYS.fetch_strategy_group, groupName],
    queryFn: () => fetchStrategyGroup(groupName.toUpperCase()) as any,
    staleTime: 0,
  });
}

export function useUncompletedTradesQuery(): UseQueryResult<Trade[], unknown> {
  return useQuery<Trade[], unknown>({
    queryKey: [QUERY_KEYS.fetch_uncompleted_trades],
    queryFn: () => fetchUncompletedTrades(),
    staleTime: 0,
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

export function useLatestBalanceSnapshot(): UseQueryResult<
  BalanceSnapshot,
  unknown
> {
  return useQuery<BalanceSnapshot, unknown>({
    queryKey: [QUERY_KEYS.fetch_latest_balance_snapshot],
    queryFn: () => fetchLatestBalanceSnapshot(),
    refetchInterval: 5000,
  });
}

export function useBalanceSnapshots1DInterval(): UseQueryResult<
  BalanceSnapshot[],
  unknown
> {
  return useQuery<BalanceSnapshot[], unknown>({
    queryKey: [QUERY_KEYS.fetch_latest_balance_snapshot],
    queryFn: () => fetchBalanceSnapshots1DInterval(),
  });
}

export function useLongshortGroup(
  groupName: string,
): UseQueryResult<LongShortGroupResponse, unknown> {
  return useQuery<LongShortGroupResponse, unknown>({
    queryKey: [QUERY_KEYS.fetch_longshort_group, groupName],
    queryFn: () => fetchLongShortGroup(groupName),
  });
}

export function useBinanceAssets(): UseQueryResult<
  BinanceUserAsset[],
  unknown
> {
  return useQuery<BinanceUserAsset[], unknown>({
    queryKey: [QUERY_KEYS.fetch_binance_user_assets],
    queryFn: () => fetchUserAssets(),
  });
}

export function useBinance24hPriceChanges(): UseQueryResult<
  BinanceTickerPriceChange[],
  unknown
> {
  return useQuery<BinanceTickerPriceChange[], unknown>({
    queryKey: [QUERY_KEYS.fetch_binance_24h_price_change],
    queryFn: () => fetchBinance24hPriceChanges(),
  });
}
