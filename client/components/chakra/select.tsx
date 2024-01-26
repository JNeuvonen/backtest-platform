import React, { CSSProperties } from "react";
import { FormControl, FormLabel, Select } from "@chakra-ui/react";

export interface SelectOption {
  label: string;
  value: string;
}

interface ChakraSelectProps {
  label: string;
  options: SelectOption[];
  id?: string;
  onChange?: (value: string) => void;
  containerStyle?: CSSProperties;
  defaultValueIndex?: number;
}

export const ChakraSelect: React.FC<ChakraSelectProps> = ({
  label,
  options,
  id,
  onChange,
  containerStyle = { maxWidth: "200px" },
  defaultValueIndex = 0,
}) => {
  if (options.length === 0) return null;
  return (
    <FormControl style={containerStyle}>
      <FormLabel htmlFor={id}>{label}</FormLabel>
      <Select
        id={id}
        onChange={(e) => onChange?.(e.target.value)}
        defaultValue={options[defaultValueIndex].value}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </Select>
    </FormControl>
  );
};
