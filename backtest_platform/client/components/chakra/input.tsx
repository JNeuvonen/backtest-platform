import React from "react";
import { FormControl, FormLabel, Input } from "@chakra-ui/react";
import { CSSProperties } from "react";

interface ChakraInputProps {
  label: string;
  id?: string;
  type?: string;
  onChange?: (value: string) => void;
  containerStyle?: CSSProperties;
  defaultValue?: string;
  placeholder?: string;
}

export const ChakraInput: React.FC<ChakraInputProps> = ({
  label,
  id,
  type = "text",
  onChange,
  containerStyle = { maxWidth: "200px" },
  defaultValue = "",
  placeholder = "",
}) => {
  return (
    <FormControl style={containerStyle}>
      <FormLabel htmlFor={id}>{label}</FormLabel>
      <Input
        id={id}
        type={type}
        onChange={(e) => onChange?.(e.target.value)}
        defaultValue={defaultValue}
        placeholder={placeholder}
      />
    </FormControl>
  );
};
