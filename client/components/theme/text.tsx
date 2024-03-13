import {
  COLOR_BRAND_SECONDARY_HIGHLIGHT,
  COLOR_CONTENT_PRIMARY,
} from "../../utils/colors";

const textClickableVariant = () => ({
  cursor: "pointer",
  _hover: {
    color: COLOR_BRAND_SECONDARY_HIGHLIGHT,
    textDecoration: "underline",
  },
});

export const textTheme = {
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
