import { extendTheme } from "@chakra-ui/react";
import {
  COLOR_BG_PRIMARY,
  COLOR_BG_PRIMARY_SHADE_THREE,
  COLOR_BRAND_SECONDARY_HIGHLIGHT,
  COLOR_BRAND_SECONDARY_SHADE_ONE,
  COLOR_CONTENT_PRIMARY,
} from "./utils/colors";

export const BUTTON_VARIANTS = {
  cta: "cta",
  grey: "grey",
  nofill: "noFill",
};

const buttonCtaVariant = () => ({
  bg: COLOR_BRAND_SECONDARY_SHADE_ONE,
  color: COLOR_CONTENT_PRIMARY,
  _hover: {
    bg: COLOR_BRAND_SECONDARY_HIGHLIGHT,
  },
});

const buttonGreyVarint = () => ({
  bg: COLOR_BG_PRIMARY,
  color: COLOR_CONTENT_PRIMARY,
  _hover: {
    bg: COLOR_BG_PRIMARY_SHADE_THREE,
  },
});

const buttonNoFillVariant = () => ({
  color: COLOR_BRAND_SECONDARY_SHADE_ONE,
  bg: "transparent",
  padding: "0px !important",
  _hover: {
    bg: "transparent",
    color: COLOR_BRAND_SECONDARY_HIGHLIGHT,
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
      noFill: () => buttonNoFillVariant(),
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
