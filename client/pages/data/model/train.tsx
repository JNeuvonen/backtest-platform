import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import { useModelQuery } from "../../../clients/queries/queries";
import { Button, Spinner } from "@chakra-ui/react";
import { useModal } from "../../../hooks/useOpen";
import { SmallTable } from "../../../components/tables/Small";
import { ToolBarStyle } from "../../../components/ToolbarStyle";
import { ChakraModal } from "../../../components/chakra/modal";
import { CreateTrainJobForm } from "../../../components/CreateTrainJobForm";

interface RouteParams {
  datasetName: string;
  modelName: string;
}

export const ModelTrainPage = () => {
  const { modelName } = usePathParams<RouteParams>();
  const { data } = useModelQuery(modelName);
  const createTrainJobModal = useModal();

  if (!data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  return (
    <div>
      <ToolBarStyle style={{ marginTop: "16px" }}>
        <Button onClick={createTrainJobModal.onOpen}>Create</Button>
      </ToolBarStyle>

      <ChakraModal
        {...createTrainJobModal}
        title="Train model"
        modalContentStyle={{ maxWidth: "80%" }}
      >
        <CreateTrainJobForm onClose={createTrainJobModal.onClose} />
      </ChakraModal>

      <SmallTable columns={[]} rows={[[]]} />
    </div>
  );
};
