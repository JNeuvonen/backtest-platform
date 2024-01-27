/* eslint-disable @typescript-eslint/no-explicit-any */

interface RequestProps {
  url: string;
  method: "GET" | "POST" | "PUT" | "DELETE";
  options?: RequestInit;
  payload?: object;
}

export interface ResponseType {
  res: any;
  status: number;
}

export interface ApiResponse {
  res: any;
  status: number;
}

export const buildRequest = async ({
  url,
  method,
  options,
  payload,
}: RequestProps): Promise<ApiResponse> => {
  try {
    if (
      payload &&
      (method === "POST" || method === "PUT" || method === "DELETE")
    ) {
      options = {
        ...options,
        headers: {
          ...options?.headers,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      };
    }
    const response = await fetch(url, {
      ...options,
      method,
    });
    const contentType = response.headers.get("content-type");
    const isJson = contentType?.includes("application/json");
    const data = isJson ? await response.json() : await response.text();
    const statusCode = response.status;
    return { res: data, status: statusCode };
  } catch (error) {
    return error;
  }
};
