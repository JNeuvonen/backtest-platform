import React from "react";
import { ChakraPopover } from "./chakra/popover";
import {
  Button,
  Heading,
  Spinner,
  Tooltip,
  useDisclosure,
} from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";
import { VscListSelection } from "react-icons/vsc";
import { useCodePresets } from "../clients/queries/queries";
import { SelectWithTextFilter } from "./SelectFilter";

interface Props {
  onPresetSelect: (selectedPreset: string) => void;
  presetCategory: string;
}

export const SelectCodePreset = (props: Props) => {
  const { onPresetSelect, presetCategory } = props;
  const popover = useDisclosure();
  const codePresetsQuery = useCodePresets();

  if (!codePresetsQuery.data) return <Spinner />;

  return (
    <ChakraPopover
      {...popover}
      setOpen={popover.onOpen}
      body={
        <div>
          <SelectWithTextFilter
            placeholder="Select"
            options={codePresetsQuery.data
              .filter((filter) => filter.category === presetCategory)
              .map((item) => {
                return {
                  label: item.name,
                  value: item.code,
                };
              })}
            onChange={(item) => {
              if (item && !Array.isArray(item) && "value" in item) {
                onPresetSelect(item.value);
                popover.onClose();
              }
            }}
            isMulti={false}
          />
        </div>
      }
      headerText={`Select ${presetCategory}`}
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
