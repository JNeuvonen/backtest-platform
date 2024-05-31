import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInputStepper,
} from "@chakra-ui/react";

export const ChakraNumberStepper = () => {
  return (
    <NumberInputStepper>
      <NumberIncrementStepper color={"white"} />
      <NumberDecrementStepper color={"white"} />
    </NumberInputStepper>
  );
};
