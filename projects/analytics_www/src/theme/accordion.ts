import { extendTheme } from "@chakra-ui/react";
import {
  COLOR_BG_PRIMARY,
  COLOR_BG_TERTIARY,
  COLOR_CONTENT_PRIMARY,
} from "./colors";

export const accordionTheme = extendTheme({
  Accordion: {
    baseStyle: {
      container: {
        bg: COLOR_BG_PRIMARY,
        color: COLOR_CONTENT_PRIMARY,
        border: COLOR_BG_TERTIARY,
      },
    },
  },
});
