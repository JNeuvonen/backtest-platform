import React, { CSSProperties } from "react";
import { Checkbox } from "@chakra-ui/react";

interface ChakraCheckboxProps {
  label: string;
  id?: string;
  isChecked: boolean;
  onChange?: (isChecked: boolean) => void;
  style?: CSSProperties;
}

export const ChakraCheckbox: React.FC<ChakraCheckboxProps> = ({
  label,
  id,
  isChecked,
  onChange,
  style,
}) => {
  return (
    <Checkbox
      id={id}
      isChecked={isChecked}
      onChange={(e) => onChange?.(e.target.checked)}
      style={style}
    >
      {label}
    </Checkbox>
  );
};
