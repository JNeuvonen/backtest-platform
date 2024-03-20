import { Button, Tooltip, useDisclosure, useToast } from "@chakra-ui/react";
import React, { useState } from "react";
import { ChakraPopover } from "./chakra/popover";
import { BUTTON_VARIANTS } from "../theme";
import { IoIosSave } from "react-icons/io";
import { ChakraInput } from "./chakra/input";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { createCodePreset } from "../clients/requests";

interface Props {
  presetCategory: string;
  onSaveCodePreset: (selectedPreset: string) => void;
  code: string;
}

export interface CodePreset {
  category: string;
  name: string;
  code: string;
}

export const SaveCodePreset = (props: Props) => {
  const { presetCategory, code } = props;
  const saveCodePresetPopover = useDisclosure();
  const [codePresetName, setCodePresetName] = useState("");
  const toast = useToast();

  const saveCodePreset = async () => {
    const body = {
      category: presetCategory,
      name: codePresetName,
      code: code,
    };

    const res = await createCodePreset(body);

    if (res.status === 200) {
      toast({
        title: "Saved code preset",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      saveCodePresetPopover.onClose();
    }
  };

  return (
    <ChakraPopover
      {...saveCodePresetPopover}
      setOpen={saveCodePresetPopover.onOpen}
      body={
        <div>
          <ChakraInput
            label="Name"
            onChange={(str) => setCodePresetName(str)}
          />
          <FormSubmitBar
            style={{ marginTop: "16px" }}
            cancelCallback={saveCodePresetPopover.onClose}
            submitCallback={saveCodePreset}
          />
        </div>
      }
      headerText="Save code preset"
      placement="bottom-start"
      useArrow={false}
    >
      <Tooltip label={"Save"}>
        <Button variant={BUTTON_VARIANTS.grey2} padding={"0px"} height={"30px"}>
          <IoIosSave color={"white"} />
        </Button>
      </Tooltip>
    </ChakraPopover>
  );
};
