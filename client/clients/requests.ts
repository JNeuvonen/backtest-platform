import { tablesUrl } from "./endpoints";
import { buildRequest } from "./fetch";

export async function fetchDatasets() {
  const url = await tablesUrl();
  return buildRequest({ method: "GET", url });
}
