import { useAuth0 } from "@auth0/auth0-react";
import { httpReq, HttpRequestOptions, HttpResponse } from "common_js";
import { useState, useEffect } from "react";

export const useAccessTokenReq = () => {
  const { getAccessTokenSilently } = useAuth0();
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    const getToken = async () => {
      try {
        const token = await getAccessTokenSilently({
          authorizationParams: {
            audience: `https://${process.env.REACT_APP_AUTH0_DOMAIN}/api/v2/`,
            scope: "read:current_user",
          },
        });
        setToken(token);
      } catch (error) {
        console.error("Error getting access token:", error);
      }
    };

    getToken();
  }, [getAccessTokenSilently]);

  const accessTokenReq = async <T = any,>(
    options: HttpRequestOptions,
  ): Promise<HttpResponse<T>> => {
    if (token) {
      options.headers = {
        ...options.headers,
        Authorization: `Bearer ${token}`,
      };
    }
    return httpReq<T>(options);
  };

  return accessTokenReq;
};