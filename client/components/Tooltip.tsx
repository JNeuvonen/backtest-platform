import React, { cloneElement, useState } from "react";
import { Tooltip } from "@chakra-ui/react";

interface Props {
  label: string;
  children: React.ReactElement;
}

export const ChakraTooltip = ({ label, children }: Props) => {
  const [isOpen, setIsOpen] = useState(false);

  const clonedChild = cloneElement(children, {
    onMouseEnter: () => setIsOpen(true),
    onMouseLeave: () => setIsOpen(false),
  });

  return (
    <Tooltip
      label={label}
      hasArrow
      isOpen={isOpen}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      {clonedChild}
    </Tooltip>
  );
};
