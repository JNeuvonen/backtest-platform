import { buildRequest } from "./fetch";

const BASE_URL = process.env.REACT_APP_BACKEND_URL;

export function fetchDatasets() {
  const url = BASE_URL + "/tables";
  return buildRequest({ method: "GET", url });
}
