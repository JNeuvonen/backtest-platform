import React from "react";
import { FormControl, FormLabel } from "@chakra-ui/react";
import { CSSProperties } from "react";

interface Props {
  children: React.ReactNode;
  label?: string | JSX.Element;
  containerStyles?: CSSProperties;
}

export const WithLabel = ({ children, label, containerStyles }: Props) => {
  return (
    <FormControl style={containerStyles}>
      <FormLabel>{label}</FormLabel>
      {children}
    </FormControl>
  );
};
