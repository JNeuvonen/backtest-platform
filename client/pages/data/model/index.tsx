import React from "react";
// import usePath from "../../../hooks/usePath";
import { ChakraModal } from "../../../components/chakra/modal";
import { useModal } from "../../../hooks/useOpen";
import { Button } from "@chakra-ui/react";
import { DatasetModelCreatePage } from "./create";

// const TAB_LABELS = ["Available", "Create", "Train"];
export const DatasetModelIndex = () => {
  // const { path } = usePath();

  const createModelModal = useModal();

  return (
    <div>
      <ChakraModal
        {...createModelModal}
        title="Create Model"
        modalContentStyle={{ maxWidth: "80%" }}
      >
        <DatasetModelCreatePage cancelCallback={createModelModal.onClose} />
      </ChakraModal>
      <Button onClick={createModelModal.onOpen}>Create</Button>
      <div>no models available</div>
    </div>
  );
};
