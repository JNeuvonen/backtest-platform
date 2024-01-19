import React, { CSSProperties } from "react";
import { FormControl, FormLabel, Select } from "@chakra-ui/react";

interface SelectOption {
  label: string;
  value: string;
}

interface ChakraSelectProps {
  label: string;
  options: SelectOption[];
  id: string;
  onChange?: (value: string) => void;
  containerStyle?: CSSProperties;
}

export const ChakraSelect: React.FC<ChakraSelectProps> = ({
  label,
  options,
  id,
  onChange,
  containerStyle = { maxWidth: "200px" },
}) => {
  return (
    <FormControl style={containerStyle}>
      <FormLabel htmlFor={id}>{label}</FormLabel>
      <Select id={id} onChange={(e) => onChange?.(e.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </Select>
    </FormControl>
  );
};
