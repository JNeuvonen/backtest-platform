const X_API_KEY = "X-API-KEY";

export const predServerHeaders = (apiKey: string) => {
  return {
    [X_API_KEY]: apiKey,
  };
};
