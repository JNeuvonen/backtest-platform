const ANALYTICS_SERV_BASE_URL = process.env.REACT_APP_ANALYTICS_SERV_URI
  ? process.env.REACT_APP_ANALYTICS_SERV_URI.replace(/\/$/, "")
  : "";

export const ANALYTICS_SERV_ROUTES = {
  v1_user: "/v1/user",
};

export const ANALYTICS_SERV_API = {
  user_from_token: () =>
    ANALYTICS_SERV_BASE_URL + ANALYTICS_SERV_ROUTES.v1_user + "/token",
};
