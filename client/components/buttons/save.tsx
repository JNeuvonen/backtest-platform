import React, { CSSProperties } from "react";
import { useAppContext } from "../../context/app";
import { KEYBIND_MSGS } from "../../utils/content";
import { ChakraTooltip } from "../Tooltip";
import { Button } from "@chakra-ui/react";
import { IoMdSave } from "react-icons/io";

interface Props {
  style?: CSSProperties;
  onClick?: () => void;
}

export const SaveButton = (props: Props) => {
  const { platform } = useAppContext();
  return (
    <ChakraTooltip label={KEYBIND_MSGS.get_save(platform)}>
      <Button
        style={{ height: "35px", marginBottom: "16px" }}
        leftIcon={<IoMdSave />}
        {...props}
      >
        Save
      </Button>
    </ChakraTooltip>
  );
};
