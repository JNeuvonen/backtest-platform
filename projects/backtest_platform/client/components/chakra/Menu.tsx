import { Menu, MenuList } from "@chakra-ui/react";
import React from "react";
import { COLOR_BG_TERTIARY } from "../../utils/colors";

interface Props {
  menuButton: JSX.Element;
  children: React.ReactNode[] | React.ReactNode;
}

export const ChakraMenu = ({ menuButton, children }: Props) => {
  return (
    <Menu>
      {menuButton}
      <MenuList border={`1px solid ${COLOR_BG_TERTIARY}`}>{children}</MenuList>
    </Menu>
  );
};
