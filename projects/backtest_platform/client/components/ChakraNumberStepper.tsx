import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInputStepper,
} from "@chakra-ui/react";
import React from "react";

export const ChakraNumberStepper = () => {
  return (
    <NumberInputStepper>
      <NumberIncrementStepper color={"white"} />
      <NumberDecrementStepper color={"white"} />
    </NumberInputStepper>
  );
};
