interface RequestProps {
  url: string;
  method: "GET" | "POST" | "PUT" | "DELETE";
  options?: RequestInit;
}

export const buildRequest = async ({ url, method, options }: RequestProps) => {
  try {
    const response = await fetch(url, { ...options, method });
    const statusCode = response.status;
    return { res: await response.json(), status: statusCode };
  } catch (error) {
    return error;
  }
};
