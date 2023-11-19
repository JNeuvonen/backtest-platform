import { extendTheme } from "@chakra-ui/react";
import {
  COLOR_BG_PRIMARY,
  COLOR_BG_PRIMARY_SHADE_THREE,
  COLOR_CONTENT_PRIMARY,
} from "./utils/colors";

const buttonCtaVariant = () => ({
  bg: COLOR_BG_PRIMARY,
  color: COLOR_BG_PRIMARY,
  _hover: {
    bg: COLOR_BG_PRIMARY,
  },
});

const buttonGreyVarint = () => ({
  bg: COLOR_BG_PRIMARY,
  color: COLOR_CONTENT_PRIMARY,
  _hover: {
    bg: COLOR_BG_PRIMARY_SHADE_THREE,
  },
});

const buttonTheme = {
  Button: {
    baseStyle: {
      padding: "4px 20px",
    },
    variants: {
      cta: () => buttonCtaVariant(),
      grey: () => buttonGreyVarint(),
    },
    defaultProps: {
      variant: "cta",
    },
  },
};

export const customChakraTheme = extendTheme({
  styles: {
    global: {
      body: {
        bg: COLOR_BG_PRIMARY,
        color: COLOR_CONTENT_PRIMARY,
      },
    },
  },
  components: {
    ...buttonTheme,
  },
});
