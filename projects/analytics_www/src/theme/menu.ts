import { menuAnatomy } from "@chakra-ui/anatomy";
import {
  COLOR_BG_PRIMARY_SHADE_ONE,
  COLOR_BG_TERTIARY,
  COLOR_CONTENT_PRIMARY,
} from "src/styles/colors";

export const menuItemTheme = {
  parts: menuAnatomy.keys,
  baseStyle: {
    item: {
      backgroundColor: COLOR_BG_PRIMARY_SHADE_ONE,
      color: COLOR_CONTENT_PRIMARY,
      _hover: {
        bg: COLOR_BG_TERTIARY,
      },
      _focus: {
        bg: COLOR_BG_TERTIARY,
      },
    },
    list: {
      border: "none",
      backgroundColor: COLOR_BG_PRIMARY_SHADE_ONE,
      borderRadius: "8px",
    },
    button: {
      fontWeight: "300",
      color: COLOR_CONTENT_PRIMARY,
      fontSize: 16,
      cursor: "pointer",
      padding: "6px",
      marginLeft: "-6px",
      borderRadius: "3px",
      _hover: {
        color: "white",
        backgroundColor: COLOR_BG_PRIMARY_SHADE_ONE,
      },
      _disabled: {
        opacity: 0.4,
        cursor: "not-allowed",
      },
    },
    command: {},
    divider: {
      my: "4",
      borderColor: "white",
      borderBottom: "2px dotted",
    },
  },
};
