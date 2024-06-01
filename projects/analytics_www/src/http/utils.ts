import { DiskManager } from "common_js";
import { toast } from "react-toastify";
import { DISK_KEYS } from "src/utils/keys";

interface ErrorOptions {
  errorNotifyUI?: boolean;
  customErrorMsgGeneratorCallback?: (error: string) => string;
  errorTitle?: string;
  onErrorCallback?: (error: string) => void;
}

export interface HttpRequestOptions {
  url: string;
  method?: "GET" | "POST" | "PUT" | "DELETE";
  data?: any;
  params?: Record<string, string>;
  headers?: Record<string, string>;
  errorOptions?: ErrorOptions;
  onSuccessCallback?: (data?: any, params?: Record<string, string>) => void;
  autoNofifyOnError?: boolean;
}

export interface HttpResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

const diskManager = new DiskManager(DISK_KEYS.access_token);

export const httpReq = async <T = any>({
  url,
  method = "GET",
  data,
  params,
  headers = {},
  errorOptions = {},
  onSuccessCallback,
  autoNofifyOnError = false,
}: HttpRequestOptions): Promise<HttpResponse<T>> => {
  const { onErrorCallback = undefined } = errorOptions;

  try {
    const urlWithParams = new URL(url);
    if (params) {
      Object.keys(params).forEach((key) =>
        urlWithParams.searchParams.append(key, params[key]),
      );
    }

    const tokenFromDisk = diskManager.read();
    const token = tokenFromDisk ? tokenFromDisk.token : null;

    const options: RequestInit = {
      method,
      headers: {
        "Content-Type": "application/json",
        ...headers,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    };

    if (data) {
      options.body = JSON.stringify(data);
    }

    const response = await fetch(urlWithParams.toString(), options);

    const contentType = response.headers.get("content-type");
    const isJson = contentType?.includes("application/json");

    if (!response.ok) {
      const errorData = isJson ? await response.json() : await response.text();
      const errorMsg = `HTTP error! Status: ${response.status}, Detail: ${
        response.statusText
      }, Error Message: ${errorData.message || ""}`;

      throw new Error(errorMsg);
    }

    const responseData: T = isJson
      ? await response.json()
      : await response.text();

    if (onSuccessCallback) {
      onSuccessCallback(data, params);
    }

    return {
      success: true,
      data: responseData,
    };
  } catch (error) {
    if (onErrorCallback) {
      onErrorCallback(error.message);
    }

    if (autoNofifyOnError) {
      toast.error(error.message, { theme: "dark" });
    }
    return {
      success: false,
      error: (error as Error).message,
    };
  }
};
