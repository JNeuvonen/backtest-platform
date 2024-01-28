import { Dataset } from "../clients/queries/response-types";

export const getDatasetColumnOptions = (dataset: Dataset) => {
  return dataset.columns.map((item) => {
    return {
      label: item,
      value: item,
    };
  });
};
