import { Dataset, DatasetMetadata } from "../clients/queries/response-types";
import { OptionType } from "../components/SelectFilter";

export const getDatasetColumnOptions = (dataset: Dataset) => {
  return dataset.columns.map((item) => {
    return {
      label: item,
      value: item,
    };
  });
};

export const getColumnOptionsAllDatasets = (datasets: DatasetMetadata[]) => {
  const ret = [] as OptionType[];

  datasets.forEach((item) => {
    const datasetName = item.table_name;
    item.columns.forEach((col) => {
      ret.push({
        value: datasetName + "_" + col,
        label: datasetName + "_" + col,
      });
    });
  });
  return ret;
};

export const convertColumnsToAgGridFormat = (
  rows: number[][],
  columns: string[]
) => {
  const ret: { [field: string]: number }[] = [];
  rows.forEach((row) => {
    const item = {};
    row.forEach((cellValue, i) => {
      item[columns[i]] = cellValue;
    });
    ret.push(item);
  });
  return ret;
};

export const createColumnChartData = (
  rows: number[],
  columnName: string,
  kline_open_time: number[],
  price_data: number[]
) => {
  const itemCount = rows.length;
  const skipItems = Math.max(1, Math.floor(itemCount / 1000));
  const ret: object[] = [];

  for (let i = 0; i < itemCount; i++) {
    if (i % skipItems === 0) {
      const item = rows[i];
      const rowObject = {};
      rowObject[columnName] = item;
      rowObject["kline_open_time"] = kline_open_time[i];
      if (price_data) {
        rowObject["price"] = price_data[i];
      } else {
        rowObject["price"] = null;
      }
      ret.push(rowObject);
    }
  }

  return ret;
};
