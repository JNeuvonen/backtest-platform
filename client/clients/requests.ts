import { URLS } from "./endpoints";
import { buildRequest } from "./fetch";

export async function fetchDatasets() {
  const url = URLS.get_tables;
  return buildRequest({ method: "GET", url });
}

export async function fetchDataset(datasetName: string) {
  const url = URLS.get_table(datasetName);
  return buildRequest({ method: "GET", url });
}

export async function fetchAllTickers() {
  const url = URLS.binance_get_all_tickers;
  return buildRequest({ method: "GET", url });
}
