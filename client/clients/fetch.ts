interface RequestProps {
  url: string;
  method: "GET" | "POST" | "PUT" | "DELETE";
  options?: RequestInit;
}

export const buildRequest = async ({ url, method, options }: RequestProps) => {
  try {
    const response = await fetch(url, { ...options, method });
    return response;
  } catch (error) {
    return error;
  }
};
