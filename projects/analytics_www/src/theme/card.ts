import { COLOR_BG_PRIMARY, COLOR_BG_PRIMARY_SHADE_TWO } from ".";

export const cardTheme = {
  Card: {
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
    variants: {
      on_modal: {
        container: {
          backgroundColor: COLOR_BG_PRIMARY_SHADE_TWO,
        },

        body: {
          backgroundColor: COLOR_BG_PRIMARY_SHADE_TWO,
        },
      },
    },
  },
};
