import React from "react";
import { Card, CardBody, CardHeader, Heading } from "@chakra-ui/react";

interface Props {
  children: React.ReactNode;
  heading?: React.ReactNode;
}
export const ChakraCard = ({ children, heading }: Props) => {
  return (
    <Card>
      {heading && <CardHeader>{heading}</CardHeader>}
      <CardBody>{children}</CardBody>
    </Card>
  );
};
