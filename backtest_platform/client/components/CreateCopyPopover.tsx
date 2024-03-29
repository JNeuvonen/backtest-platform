import { Button, Input, Spinner, useToast } from "@chakra-ui/react";
import React, { useState } from "react";
import { BUTTON_VARIANTS } from "../theme";
import { ToolBarStyle } from "./ToolbarStyle";
import { WithLabel } from "./form/WithLabel";
import { useKeyListener } from "../hooks/useKeyListener";
import { createCopyOfDataset } from "../clients/requests";

interface Props {
  datasetName: string;
  successCallback: () => void;
  cancelCallback: () => void;
}

export const CreateCopyPopover = ({
  cancelCallback,
  datasetName,
  successCallback,
}: Props) => {
  const [copyName, setCopyName] = useState(datasetName);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const submit = async () => {
    setIsLoading(true);
    const res = await createCopyOfDataset(datasetName, copyName);
    if (res.status === 200) {
      toast({
        title: "Created copy of the dataset",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      successCallback();
      setIsLoading(false);
    } else {
      setIsLoading(false);
    }
  };

  useKeyListener({
    eventAction: (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        if (copyName === "" || copyName === datasetName) return;
        submit();
      }
    },
  });

  if (isLoading) {
    return <Spinner />;
  }
  return (
    <div>
      <WithLabel label={"Set name of the copy"}>
        <Input
          value={copyName}
          onChange={(e) => {
            const validChars = /^[A-Za-z0-9_ ]*$/;
            if (validChars.test(e.target.value)) {
              setCopyName(e.target.value);
            }
          }}
        />
      </WithLabel>
      <ToolBarStyle style={{ marginTop: "8px" }}>
        <Button
          height={"24px"}
          variant={BUTTON_VARIANTS.grey}
          onClick={cancelCallback}
        >
          Cancel
        </Button>
        <Button
          height={"24px"}
          isDisabled={copyName === "" || copyName === datasetName}
          onClick={submit}
        >
          Submit
        </Button>
      </ToolBarStyle>
    </div>
  );
};
