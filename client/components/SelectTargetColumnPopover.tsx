import React, { CSSProperties } from "react";
import { OptionType, SelectWithTextFilter } from "./SelectFilter";
import { MultiValue, SingleValue } from "react-select";

interface Props {
  options: OptionType[];
  placeholder: string;
  selectCallback: (value: string) => void;
  containerStyles?: CSSProperties;
}

export const SelectColumnPopover = ({
  options,
  placeholder,
  selectCallback,
  containerStyles,
}: Props) => {
  return (
    <SelectWithTextFilter
      options={options}
      isMulti={false}
      placeholder={placeholder}
      containerStyle={containerStyles}
      onChange={(
        selectedOptions: SingleValue<OptionType> | MultiValue<OptionType>
      ) => {
        const option = selectedOptions as SingleValue<OptionType>;
        if (option) selectCallback(option.value as string);
      }}
    />
  );
};
