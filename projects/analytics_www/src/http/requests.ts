import { BinanceSymbolPrice, SPOT_MARKET_INFO_ENDPOINT } from "common_js";
import { httpReq, ANALYTICS_SERV_API } from ".";
import { HttpRequestOptions } from "./utils";

export const fetchUserUsingToken = async () => {
  const res = await httpReq({
    url: ANALYTICS_SERV_API.user_from_token(),
  });
  return res;
};

export const fetchUserParams = (): HttpRequestOptions => {
  return {
    url: ANALYTICS_SERV_API.user_from_token(),
  };
};

export const fetchBalanceSnapshotsOptions = (): HttpRequestOptions => {
  return {
    url: ANALYTICS_SERV_API.read_balance_snapshots(),
  };
};

export const fetchBalanceSnapshots = async () => {
  try {
    const res = await httpReq({ ...fetchBalanceSnapshotsOptions() });

    if (res.success) {
      return res.data.data;
    }
    return [];
  } catch {
    return [];
  }
};

export const fetchStrategies = async () => {
  try {
    const res = await httpReq({ url: ANALYTICS_SERV_API.fetch_strategies() });

    if (res.success) {
      return res.data;
    }
    return {};
  } catch {
    return {};
  }
};

export const fetchStrategyGroup = async (groupName: string) => {
  try {
    const res = await httpReq({
      url: ANALYTICS_SERV_API.fetch_strategy_group(groupName),
    });
    if (res.success) {
      return res.data;
    }
    return {};
  } catch {
    return {};
  }
};

export const fetchBinancePriceInfo = async () => {
  try {
    const response = await fetch(SPOT_MARKET_INFO_ENDPOINT);
    if (response.ok) {
      const data: BinanceSymbolPrice[] = await response.json();
      return data;
    }
    return [] as BinanceSymbolPrice[];
  } catch {
    return [] as BinanceSymbolPrice[];
  }
};

export const fetchLatestBalanceSnapshot = async () => {
  try {
    const res = await httpReq({
      url: ANALYTICS_SERV_API.fetch_balance_snapshot_latest(),
    });
    if (res.success) {
      return res.data.data;
    }
    return {};
  } catch {
    return {};
  }
};

export const fetchBalanceSnapshots1DInterval = async () => {
  try {
    const res = await httpReq({
      url: ANALYTICS_SERV_API.fetch_balance_snapshot_1d_interval(),
    });
    if (res.success) {
      return res.data;
    }
    return {};
  } catch {
    return {};
  }
};

export const fetchLongShortGroup = async (groupName: string) => {
  try {
    const res = await httpReq({
      url: ANALYTICS_SERV_API.fetch_longshort_group(groupName),
    });
    if (res.success) {
      return res.data;
    }
    return {};
  } catch {
    return {};
  }
};

export const updateManyStrategies = async (payload) => {
  const res = await httpReq({
    url: ANALYTICS_SERV_API.update_many_strateies(),
    data: payload,
    method: "PUT",
    autoNofifyOnError: true,
  });
  return res;
};
