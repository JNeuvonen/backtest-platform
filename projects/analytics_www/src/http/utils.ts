interface ErrorOptions {
  errorNotifyUI?: boolean;
  customErrorMsgGeneratorCallback?: (error: string) => string;
  errorTitle?: string;
  onErrorCallback?: () => void;
}

export interface HttpRequestOptions {
  url: string;
  method?: "GET" | "POST" | "PUT" | "DELETE";
  data?: any;
  params?: Record<string, string>;
  headers?: Record<string, string>;
  errorOptions?: ErrorOptions;
  onSuccessCallback?: (data?: any, params?: Record<string, string>) => void;
}

export interface HttpResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

export const httpReq = async <T = any>({
  url,
  method = "GET",
  data,
  params,
  headers = {},
  errorOptions = {},
  onSuccessCallback,
}: HttpRequestOptions): Promise<HttpResponse<T>> => {
  const { onErrorCallback = undefined } = errorOptions;

  try {
    const urlWithParams = new URL(url);
    if (params) {
      Object.keys(params).forEach((key) =>
        urlWithParams.searchParams.append(key, params[key]),
      );
    }

    const options: RequestInit = {
      method,
      headers: {
        "Content-Type": "application/json",
        ...headers,
      },
    };

    if (data) {
      options.body = JSON.stringify(data);
    }

    const response = await fetch(urlWithParams.toString(), options);

    if (!response.ok) {
      const errorData = await response.json();
      const errorMsg = `HTTP error! Status: ${response.status}, Detail: ${
        response.statusText
      }, Error Message: ${errorData.message || ""}`;

      throw new Error(errorMsg);
    }

    const responseData: T = await response.json();

    if (onSuccessCallback) {
      onSuccessCallback(data, params);
    }

    return {
      success: true,
      data: responseData,
    };
  } catch (error) {
    if (onErrorCallback) {
      onErrorCallback();
    }
    return {
      success: false,
      error: (error as Error).message,
    };
  }
};
