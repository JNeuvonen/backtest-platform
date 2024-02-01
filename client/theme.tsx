import { extendTheme } from "@chakra-ui/react";
import { menuAnatomy } from "@chakra-ui/anatomy";
import {
  COLOR_BG_PRIMARY,
  COLOR_BG_PRIMARY_SHADE_THREE,
  COLOR_BG_PRIMARY_SHADE_TWO,
  COLOR_BG_SECONDARY,
  COLOR_BG_TERTIARY,
  COLOR_BRAND_SECONDARY_HIGHLIGHT,
  COLOR_BRAND_SECONDARY_SHADE_ONE,
  COLOR_CONTENT_PRIMARY,
} from "./utils/colors";

export const BUTTON_VARIANTS = {
  cta: "cta",
  grey: "grey",
  nofill: "noFill",
  grey2: "grey2",
};

export const TEXT_VARIANTS = {
  clickable: "clickable",
};

const buttonCtaVariant = () => ({
  bg: COLOR_BRAND_SECONDARY_SHADE_ONE,
  color: COLOR_CONTENT_PRIMARY,
  _hover: {
    bg: COLOR_BRAND_SECONDARY_HIGHLIGHT,
  },
});

const buttonGreyVariant = () => ({
  bg: COLOR_BG_PRIMARY,
  color: COLOR_CONTENT_PRIMARY,
  _hover: {
    bg: COLOR_BG_PRIMARY_SHADE_THREE,
  },
});

const buttonGrey2Variant = () => ({
  bg: COLOR_BG_TERTIARY,
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
      grey: () => buttonGreyVariant(),
      noFill: () => buttonNoFillVariant(),
      grey2: () => buttonGrey2Variant(),
    },
    defaultProps: {
      variant: "cta",
    },
  },
};

const textClickableVariant = () => ({
  cursor: "pointer",
  _hover: {
    color: COLOR_BRAND_SECONDARY_HIGHLIGHT,
    textDecoration: "underline",
  },
});

const textTheme = {
  Text: {
    baseStyle: {
      color: COLOR_CONTENT_PRIMARY,
    },
    variants: {
      clickable: () => textClickableVariant(),
    },
    defaultProps: {
      variant: "",
    },
  },
};

const headingTheme = {
  Heading: {
    baseStyle: {
      color: COLOR_CONTENT_PRIMARY,
    },
    sizes: {},
    variants: {},
    defaultProps: {},
  },
};

const cardTheme = {
  parts: ["container", "header", "body"],
  baseStyle: {
    container: {
      backgroundColor: COLOR_BG_PRIMARY,
    },
    header: {},
    body: {
      backgroundColor: COLOR_BG_PRIMARY,
    },
  },
};
const menuItemTheme = {
  parts: menuAnatomy.keys,
  baseStyle: {
    item: {
      backgroundColor: COLOR_BG_PRIMARY_SHADE_TWO,
      color: COLOR_CONTENT_PRIMARY,
      _hover: {
        bg: COLOR_BG_SECONDARY,
      },
      _focus: {
        bg: COLOR_BG_SECONDARY,
      },
    },
    list: {
      border: "none",

      backgroundColor: COLOR_BG_PRIMARY_SHADE_TWO,
    },
    button: {
      // this will style the MenuButton component
      fontWeight: "medium",
      bg: "teal.500",
      color: "gray.200",
      _hover: {
        bg: "teal.600",
        color: "white",
      },
    },
    command: {},
    divider: {
      // this will style the MenuDivider component
      my: "4",
      borderColor: "white",
      borderBottom: "2px dotted",
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
    ...textTheme,
    ...headingTheme,
    ...cardTheme,
    Card: cardTheme,
    Menu: menuItemTheme,
  },
});
