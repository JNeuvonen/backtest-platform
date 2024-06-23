import { extendTheme } from "@chakra-ui/react";
import {
  accordionTheme,
  COLOR_BG_PRIMARY_SHADE_ONE,
  COLOR_CONTENT_PRIMARY,
} from ".";
import { buttonTheme } from "./button";
import { cardTheme } from "./card";
import { headingTheme } from "./heading";
import { menuItemTheme } from "./menu";

export const customChakraTheme = extendTheme({
  styles: {
    global: {
      body: {
        bg: COLOR_BG_PRIMARY_SHADE_ONE,
        color: COLOR_CONTENT_PRIMARY,
      },
    },
  },
  components: {
    ...accordionTheme,
    ...headingTheme,
    ...cardTheme,
    ...buttonTheme,
    Menu: menuItemTheme,
  },
});
