import React, { useState } from "react";
import { DatasetMetadata } from "../clients/queries/response-types";
import { OptionType, SelectWithTextFilter } from "./SelectFilter";
import { MultiValue, SingleValue } from "react-select";
import { FormSubmitBar } from "./form/CancelSubmitBar";
import { useKeyListener } from "../hooks/useKeyListener";

interface Props {
  onSelect: (
    selectedItems: SingleValue<OptionType> | MultiValue<OptionType>
  ) => void;
  cancelCallback: () => void;
  datasets: DatasetMetadata[];
}

export const SelectDataset = ({
  onSelect,
  datasets,
  cancelCallback,
}: Props) => {
  const [selectedOption, setSelectedOption] =
    useState<SingleValue<OptionType>>(null);
  const getOptions = () => {
    return datasets.map((item) => {
      return {
        value: item.table_name,
        label: item.table_name,
      };
    });
  };
  useKeyListener({
    eventAction: (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        if (selectedOption !== null) {
          onSelect(selectedOption);
        }
      }
    },
  });

  const submit = () => {
    onSelect(selectedOption);
  };
  return (
    <div>
      <SelectWithTextFilter
        options={getOptions()}
        isMulti={false}
        placeholder="Select dataset"
        onChange={(selectedOption) => {
          setSelectedOption(selectedOption as SingleValue<OptionType>);
        }}
      />
      <FormSubmitBar
        style={{ marginTop: "16px" }}
        submitCallback={submit}
        submitText="Ok"
        submitDisabled={selectedOption === null}
        cancelCallback={cancelCallback}
      />
    </div>
  );
};
