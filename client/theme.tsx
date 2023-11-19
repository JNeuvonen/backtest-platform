import { extendTheme } from "@chakra-ui/react";

const COLORS = {
  action_button_blue: {
    default: "#167BED",
    hover: "#024ca1",
    text_color: "white",
  },
  button_grey: {
    default: "#3B3B3B",
    hover: "#525252",
    text_color: "white",
  },
  body: {
    bg: "#212121",
  },
};

const buttonCtaVariant = () => ({
  bg: COLORS.action_button_blue.default,
  color: COLORS.action_button_blue.text_color,
  _hover: {
    bg: COLORS.action_button_blue.hover,
  },
});

const buttonGreyVarint = () => ({
  bg: COLORS.button_grey.default,
  color: COLORS.button_grey.text_color,
  _hover: {
    bg: COLORS.button_grey.hover,
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

const tableTheme = {
  Table: {
    baseStyle: {
      // You can also style td or other parts of the table if required
      bgColor: "red",
    },
    variants: {
      default: {
        th: {
          borderColor: "red",
          borderBottom: "1px solid",
        },
      },
    },
    defaultProps: {
      variant: "default",
    },
  },
};

export const customChakraTheme = extendTheme({
  styles: {
    global: {
      body: {
        bg: COLORS.body.bg,
        color: "#f0f0f0",
      },
    },
  },
  components: {
    ...buttonTheme,
    ...tableTheme,
  },
});
