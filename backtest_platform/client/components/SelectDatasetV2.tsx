import React from "react";
import { DatasetMetadata } from "../clients/queries/response-types";
import { OptionType, SelectWithTextFilter } from "./SelectFilter";
import { MultiValue, SingleValue } from "react-select";

interface Props {
  onSelect: (
    selectedOptions: MultiValue<OptionType> | SingleValue<OptionType>
  ) => void;
  datasets: DatasetMetadata[];
  multiSelect?: boolean;
  value: MultiValue<OptionType> | SingleValue<OptionType>;
}

export const SelectDatasetV2 = ({
  onSelect,
  datasets,
  multiSelect = false,
  value,
}: Props) => {
  const getOptions = () => {
    return datasets.map((item) => {
      return {
        value: item.table_name,
        label: item.table_name,
      };
    });
  };
  return (
    <div>
      <SelectWithTextFilter
        options={getOptions()}
        isMulti={multiSelect}
        value={value}
        placeholder="Select dataset"
        onChange={(selectedOption) => {
          onSelect(selectedOption);
        }}
      />
    </div>
  );
};
