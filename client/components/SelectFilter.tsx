import React, { CSSProperties } from "react";
import Select, { MultiValue, SingleValue } from "react-select";
import {
  COLOR_BG_PRIMARY_SHADE_ONE,
  COLOR_BG_SECONDARY_SHADE_THREE,
} from "../utils/colors";
import { FormControl, FormLabel } from "@chakra-ui/react";

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
  label?: string;
  containerStyle?: CSSProperties;
  id?: string;
}

const SearchSelect = ({
  options,
  onChange,
  isMulti,
  closeMenuOnSelect,
  placeholder,
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

export const SelectWithTextFilter: React.FC<SelectWithTextFilterProps> = ({
  options,
  onChange,
  placeholder,
  isMulti,
  closeMenuOnSelect = true,
  label = "",
  containerStyle = {},
  id = "",
}) => {
  if (label) {
    return (
      <FormControl style={containerStyle}>
        <FormLabel htmlFor={id}>{label}</FormLabel>
        <SearchSelect
          options={options}
          onChange={onChange}
          placeholder={placeholder}
          isMulti={isMulti}
          closeMenuOnSelect={closeMenuOnSelect}
        />
      </FormControl>
    );
  }

  return (
    <SearchSelect
      options={options}
      onChange={onChange}
      placeholder={placeholder}
      isMulti={isMulti}
      closeMenuOnSelect={closeMenuOnSelect}
    />
  );
};
