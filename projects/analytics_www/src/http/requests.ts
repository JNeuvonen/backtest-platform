import { httpReq, ANALYTICS_SERV_API, HttpRequestOptions } from "common_js";

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
