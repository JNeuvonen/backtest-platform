const ANALYTICS_SERV_BASE_URL = process.env.REACT_APP_ANALYTICS_SERV_URI
  ? process.env.REACT_APP_ANALYTICS_SERV_URI.replace(/\/$/, "")
  : "";

export const ANALYTICS_SERV_ROUTES = {
  v1_user: "/v1/user",
  v1_balance_snapshot: "/v1/balance-snapshot",
  v1_strategy: "/v1/strategy",
};

export const ANALYTICS_SERV_API = {
  user_from_token: () =>
    ANALYTICS_SERV_BASE_URL + ANALYTICS_SERV_ROUTES.v1_user + "/token",
  read_balance_snapshots: () =>
    ANALYTICS_SERV_BASE_URL + ANALYTICS_SERV_ROUTES.v1_balance_snapshot + "/",
  fetch_strategies: () =>
    ANALYTICS_SERV_BASE_URL + ANALYTICS_SERV_ROUTES.v1_strategy + "/",
  fetch_strategy_group: (groupName: string) =>
    ANALYTICS_SERV_BASE_URL +
    ANALYTICS_SERV_ROUTES.v1_strategy +
    `/strategy-group/${groupName}`,
  fetch_balance_snapshot_latest: () =>
    ANALYTICS_SERV_BASE_URL +
    ANALYTICS_SERV_ROUTES.v1_balance_snapshot +
    "/latest",
  fetch_balance_snapshot_1d_interval: () =>
    ANALYTICS_SERV_BASE_URL +
    ANALYTICS_SERV_ROUTES.v1_balance_snapshot +
    "/1d-interval",
  update_many_strateies: () =>
    ANALYTICS_SERV_BASE_URL +
    ANALYTICS_SERV_ROUTES.v1_strategy +
    "/update-many",
};
