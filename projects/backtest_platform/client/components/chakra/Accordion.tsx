import React, { ReactNode } from "react";
import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  AccordionProps,
} from "@chakra-ui/react";

interface CustomAccordionProps extends AccordionProps {
  items: {
    title: ReactNode;
    content: ReactNode;
  }[];
}

export const ChakraAccordion: React.FC<CustomAccordionProps> = ({
  items,
  ...props
}) => {
  return (
    <Accordion {...props}>
      {items.map((item, index) => (
        <AccordionItem key={index}>
          <AccordionButton>
            <h2>{item.title}</h2>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel>{item.content}</AccordionPanel>
        </AccordionItem>
      ))}
    </Accordion>
  );
};
