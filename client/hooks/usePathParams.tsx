import { useParams } from "react-router-dom";

type PathParams<T = {}> = {
  [key in keyof T]: string;
};

export const usePathParams = <T extends {} = {}>() => {
  const params = useParams<PathParams<T>>();

  const transformedParams = Object.keys(params).reduce((acc, key) => {
    const newKey = key.replace(":", "") as keyof T;
    acc[newKey] = params[key];
    return acc;
  }, {} as PathParams<T>);

  return transformedParams;
};
