import React from "react";
import Select, { MultiValue, SingleValue } from "react-select";
import {
  COLOR_BG_PRIMARY_SHADE_ONE,
  COLOR_BG_SECONDARY_SHADE_THREE,
} from "../utils/colors";

export interface OptionType {
  value: string;
  label: string;
}

interface SelectWithTextFilterProps {
  options: OptionType[];
  onChange: (
    selectedOptions: MultiValue<OptionType> | SingleValue<OptionType>
  ) => void;
  placeholder?: string;
  isMulti: boolean;
  closeMenuOnSelect?: boolean;
}

export const SelectWithTextFilter: React.FC<SelectWithTextFilterProps> = ({
  options,
  onChange,
  placeholder,
  isMulti,
  closeMenuOnSelect = true,
}) => {
  return (
    <Select
      options={options}
      onChange={onChange}
      isMulti={isMulti}
      isSearchable={true}
      closeMenuOnSelect={closeMenuOnSelect}
      placeholder={placeholder || "Select an option"}
      styles={{
        control: (provided) => {
          return {
            ...provided,
            background: COLOR_BG_SECONDARY_SHADE_THREE,
            borderColor: COLOR_BG_PRIMARY_SHADE_ONE,
            color: "white",
          };
        },
        option: (provided) => {
          return {
            ...provided,
            color: "black",
          };
        },
      }}
    />
  );
};
