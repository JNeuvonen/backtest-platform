import React, { CSSProperties } from "react";
import { Card, CardBody, CardHeader } from "@chakra-ui/react";

interface Props {
  children: React.ReactNode;
  heading?: React.ReactNode;
  variant?: string;
  containerStyles?: CSSProperties;
}
export const ChakraCard = ({
  children,
  heading,
  variant,
  containerStyles,
}: Props) => {
  return (
    <Card variant={variant} style={containerStyles}>
      {heading && <CardHeader>{heading}</CardHeader>}
      <CardBody>{children}</CardBody>
    </Card>
  );
};
