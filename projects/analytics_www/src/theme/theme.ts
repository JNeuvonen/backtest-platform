import { extendTheme } from "@chakra-ui/react";
import { COLOR_BG_PRIMARY_SHADE_ONE, COLOR_CONTENT_PRIMARY } from ".";
import { cardTheme } from "./card";
import { headingTheme } from "./heading";

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
    ...headingTheme,
    ...cardTheme,
  },
});
