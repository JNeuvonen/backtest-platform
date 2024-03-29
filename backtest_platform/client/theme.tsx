import { extendTheme } from "@chakra-ui/react";
import { COLOR_BG_PRIMARY, COLOR_CONTENT_PRIMARY } from "./utils/colors";
import {
  cardTheme,
  menuItemTheme,
  headingTheme,
  textTheme,
  buttonTheme,
} from "./components/theme";

export const BUTTON_VARIANTS = {
  cta: "cta",
  grey: "grey",
  nofill: "noFill",
  grey2: "grey2",
};

export const CARD_VARIANTS = {
  on_modal: "on_modal",
};

export const TEXT_VARIANTS = {
  clickable: "clickable",
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
