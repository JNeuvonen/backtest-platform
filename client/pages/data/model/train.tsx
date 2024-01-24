import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useModelQuery,
  useModelTrainMetadata,
} from "../../../clients/queries/queries";
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
  const { data: modelData } = useModelQuery(modelName);
  const { data: allTrainingMetadata } = useModelTrainMetadata(modelName);
  const createTrainJobModal = useModal();

  if (!modelData || !allTrainingMetadata) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  console.log(allTrainingMetadata);

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
