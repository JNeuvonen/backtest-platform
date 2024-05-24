import React from "react";
import { StatHelpText, StatArrow } from "@chakra-ui/react";

interface ChakraStatHelpTextProps {
  num: number;
  percentage?: boolean;
}

const ChakraStatHelpText: React.FC<ChakraStatHelpTextProps> = ({
  num,
  percentage = true,
}) => {
  if (num === 0) {
    return null;
  }

  return (
    <StatHelpText>
      <StatArrow type={num > 0 ? "increase" : "decrease"} />
      {Math.abs(num)}
      {percentage ? "%" : null}
    </StatHelpText>
  );
};

export default ChakraStatHelpText;
