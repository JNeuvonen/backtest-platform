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
