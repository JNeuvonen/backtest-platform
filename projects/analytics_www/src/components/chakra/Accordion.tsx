import React, { CSSProperties } from "react";
import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Box,
} from "@chakra-ui/react";
import { COLOR_BRAND_SECONDARY_SHADE_ONE } from "src/theme";

interface Props {
  children: React.ReactNode;
  heading?: React.ReactNode;
  variant?: string;
  containerStyles?: CSSProperties;
}

export const ChakraAccordion = ({
  children,
  heading,
  containerStyles,
}: Props) => {
  return (
    <Accordion allowToggle style={{ ...containerStyles }}>
      <AccordionItem>
        <h2>
          <AccordionButton>
            {heading && (
              <Box
                flex="1"
                textAlign="left"
                color={COLOR_BRAND_SECONDARY_SHADE_ONE}
              >
                {heading}
              </Box>
            )}
            <AccordionIcon color={COLOR_BRAND_SECONDARY_SHADE_ONE} />
          </AccordionButton>
        </h2>
        <AccordionPanel pb={4}>{children}</AccordionPanel>
      </AccordionItem>
    </Accordion>
  );
};
