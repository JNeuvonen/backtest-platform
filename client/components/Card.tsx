import React, { ReactNode } from "react";
import { Box, BoxProps } from "@chakra-ui/react";
import {
  COLOR_BG_PRIMARY_SHADE_TWO,
  COLOR_DARK_BG_PRIMARY_SHADE_THREE,
} from "../utils/colors";

interface BasicCardProps extends BoxProps {
  children: ReactNode;
}

export const BasicCard: React.FC<BasicCardProps> = ({ children, ...props }) => {
  return (
    <Box
      boxShadow="md"
      borderRadius="lg"
      {...props}
      bg={COLOR_BG_PRIMARY_SHADE_TWO}
      _hover={{
        bg: COLOR_DARK_BG_PRIMARY_SHADE_THREE,
        cursor: "pointer",
      }}
    >
      {children}
    </Box>
  );
};
