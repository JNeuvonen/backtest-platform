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

export async function fetchColumn(datasetName: string, columnName: string) {
  const url = URLS.get_column(datasetName, columnName);
  return buildRequest({ method: "GET", url });
}

export async function fetchAllTickers() {
  const url = URLS.binance_get_all_tickers;
  return buildRequest({ method: "GET", url });
}

export async function renameColumnName(
  datasetName: string,
  oldName: string,
  newName: string,
  isTimeseriesCol: boolean
) {
  const url = URLS.rename_column(datasetName);
  return buildRequest({
    method: "POST",
    url,
    payload: {
      old_col_name: oldName,
      new_col_name: newName,
      is_timeseries_col: isTimeseriesCol,
    },
  });
}
