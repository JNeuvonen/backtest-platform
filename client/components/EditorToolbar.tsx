import React from "react";
import { Button } from "@chakra-ui/react";
import { useAppContext } from "../context/app";
import { KEYBIND_MSGS, getParenthesisSize } from "../utils/content";
import { ChakraTooltip } from "./Tooltip";
import { useEditorContext } from "../context/editor";
import { BUTTON_VARIANTS } from "../theme";

export const EditorToolbar = () => {
  const { platform } = useAppContext();
  const {
    isDelMode,
    isSaveDisabled,
    addModalSetOpen,
    isDeleteDisabled,
    deleteIsLoading,
    deleteColumns,
    delModalSetOpen,
  } = useEditorContext();

  return (
    <div style={{ display: "flex", gap: "8px" }}>
      <ChakraTooltip label={KEYBIND_MSGS.get_save(platform)}>
        <Button
          style={{ height: "35px", marginBottom: "16px" }}
          isDisabled={isSaveDisabled()}
          onClick={() => addModalSetOpen(true)}
        >
          Save
        </Button>
      </ChakraTooltip>

      {isDelMode && (
        <ChakraTooltip label={KEYBIND_MSGS.get_save(platform)}>
          <Button
            style={{ height: "35px", marginBottom: "16px" }}
            isDisabled={isDeleteDisabled()}
            onClick={() => delModalSetOpen(true)}
            variant={BUTTON_VARIANTS.grey2}
            isLoading={deleteIsLoading}
          >
            Delete {getParenthesisSize(deleteColumns.length)}
          </Button>
        </ChakraTooltip>
      )}
    </div>
  );
};
