import React from "react";
import { ChakraPopover } from "./chakra/popover";
import { Button, Tooltip, useDisclosure } from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";
import { VscListSelection } from "react-icons/vsc";

interface Props {
  onPresetSelect: (selectedPreset: string) => void;
  presetCategory: string;
}

export const SelectCodePreset = (props: Props) => {
  const { onPresetSelect } = props;
  const longCondPresetPopover = useDisclosure();

  return (
    <ChakraPopover
      {...longCondPresetPopover}
      setOpen={longCondPresetPopover.onOpen}
      body={<div>Code preset test</div>}
      headerText="Select code preset"
      placement="bottom-start"
      useArrow={false}
    >
      <Tooltip label={"Select code preset"}>
        <Button variant={BUTTON_VARIANTS.grey2} padding={"0px"} height={"30px"}>
          <VscListSelection color={"white"} />
        </Button>
      </Tooltip>
    </ChakraPopover>
  );
};
