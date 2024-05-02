import React from "react";
import { useCodePresets } from "../clients/queries/queries";
import { Button, Tooltip, useDisclosure } from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";
import { IoMdSettings } from "react-icons/io";
import { ChakraModal } from "./chakra/modal";

interface Props {
  onPresetSelect: (selectedPreset: string) => void;
  presetCategory: string;
}

export const ManagePresets = ({ onPresetSelect, presetCategory }) => {
  const codePresetsQuery = useCodePresets();
  const managePresetsModal = useDisclosure();

  return (
    <div>
      <Tooltip label={"Manage code presets"}>
        <Button
          variant={BUTTON_VARIANTS.grey2}
          padding={"0px"}
          height={"30px"}
          onClick={managePresetsModal.onOpen}
        >
          <IoMdSettings />
        </Button>
      </Tooltip>
      <ChakraModal
        {...managePresetsModal}
        title={"Code presets"}
        modalContentStyle={{ maxWidth: "70%", marginTop: "5%" }}
      >
        <div></div>
      </ChakraModal>
    </div>
  );
};
