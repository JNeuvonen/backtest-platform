import React from "react";
import { Card, CardBody, CardHeader } from "@chakra-ui/react";

interface Props {
  children: React.ReactNode;
  heading?: React.ReactNode;
  variant?: string;
}
export const ChakraCard = ({ children, heading, variant }: Props) => {
  return (
    <Card variant={variant}>
      {heading && <CardHeader>{heading}</CardHeader>}
      <CardBody>{children}</CardBody>
    </Card>
  );
};
