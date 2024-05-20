/* eslint-disable */
import { useLocation } from "react-router-dom";

type QueryParams<T = {}> = {
  [key in keyof T]: string | undefined;
};

const useQueryParams = <T extends QueryParams<T> = {}>() => {
  const { search } = useLocation();
  const queryParams = new URLSearchParams(search);

  const params: QueryParams<T> = {} as QueryParams<T>;
  queryParams.forEach((value, key) => {
    (params as any)[key] = value;
  });

  return params as QueryParams<T>;
};

export default useQueryParams;
