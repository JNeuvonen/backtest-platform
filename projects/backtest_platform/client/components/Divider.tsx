import React from "react";
import { CSSProperties } from "react";
import { COLOR_DARK_BG_TERTIARY_SHADE_THREE } from "../utils/colors";
import { Divider } from "@chakra-ui/react";

interface Props {
  style?: CSSProperties;
  borderColor?: string;
}

export const ChakraDivider = ({
  style,
  borderColor = COLOR_DARK_BG_TERTIARY_SHADE_THREE,
}: Props) => {
  return <Divider style={style} borderColor={borderColor} />;
};
