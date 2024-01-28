import React from "react";
import { OptionType, SelectWithTextFilter } from "./SelectFilter";
import { MultiValue, SingleValue } from "react-select";

interface Props {
  options: OptionType[];
  placeholder: string;
  selectCallback: (value: string) => void;
}

export const SelectColumnPopover = ({
  options,
  placeholder,
  selectCallback,
}: Props) => {
  return (
    <SelectWithTextFilter
      options={options}
      isMulti={false}
      placeholder={placeholder}
      onChange={(
        selectedOptions: SingleValue<OptionType> | MultiValue<OptionType>
      ) => {
        const option = selectedOptions as SingleValue<OptionType>;
        if (option) selectCallback(option.value as string);
      }}
    />
  );
};
