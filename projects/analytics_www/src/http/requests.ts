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
