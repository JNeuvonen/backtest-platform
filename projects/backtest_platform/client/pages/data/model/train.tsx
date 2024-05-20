import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useModelQuery,
  useModelTrainMetadata,
} from "../../../clients/queries/queries";
import { Button, Checkbox, Spinner } from "@chakra-ui/react";
import { useModal } from "../../../hooks/useOpen";
import { SmallTable } from "../../../components/tables/Small";
import { ToolBarStyle } from "../../../components/ToolbarStyle";
import { ChakraModal } from "../../../components/chakra/modal";
import { CreateTrainJobForm } from "../../../components/CreateTrainJobForm";
import { MdPlayArrow } from "react-icons/md";
import { MdOutlinePause } from "react-icons/md";
import { BUTTON_VARIANTS } from "../../../theme";
import { useMessageListener } from "../../../hooks/useMessageListener";
import { DOM_EVENT_CHANNELS } from "../../../utils/constants";
import { Link } from "react-router-dom";
import { getTrainJobPath } from "../../../utils/navigate";

interface RouteParams {
  datasetName: string;
  modelId: string;
}

const COLUMNS = [
  "Nr",
  "Epochs",
  "Is training",
  "Backtest on validation set",
  "Save weights on every epoch",
  "Control",
];

export const ModelTrainPage = () => {
  const { modelId, datasetName } = usePathParams<RouteParams>();
  const { data: modelData, refetch: refetchModelData } = useModelQuery(modelId);
  const { data: allTrainingMetadata, refetch: refetchAllTrainingMetadata } =
    useModelTrainMetadata(modelId);
  const createTrainJobModal = useModal();

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_component,
    messageCallback: () => {
      refetchModelData();
      refetchAllTrainingMetadata();
    },
  });

  if (!modelData || !allTrainingMetadata) {
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

      <SmallTable
        columns={COLUMNS}
        rows={allTrainingMetadata.map((item, i) => {
          return [
            <Link
              key={-1}
              className="link-default"
              to={getTrainJobPath(
                datasetName,
                Number(modelId),
                String(item.train.id)
              )}
            >
              {i + 1}
            </Link>,
            `${item.train.epochs_ran}/${item.train.num_epochs}`,
            <Checkbox
              isChecked={item.train.is_training}
              disabled={true}
              key={0}
            />,
            <Checkbox
              isChecked={item.train.backtest_on_validation_set}
              disabled={true}
              key={1}
            />,
            <Checkbox
              isChecked={item.train.save_model_every_epoch}
              disabled={true}
              key={2}
            />,
            <div style={{ display: "flex", gap: "4px" }} key={3}>
              {item.train.is_training ? (
                <Button
                  leftIcon={<MdOutlinePause />}
                  variant={BUTTON_VARIANTS.grey}
                  style={{ height: "28px" }}
                  key={4}
                >
                  Pause
                </Button>
              ) : (
                <Button
                  leftIcon={<MdPlayArrow />}
                  variant={BUTTON_VARIANTS.grey}
                  style={{ height: "28px" }}
                  key={5}
                >
                  Train
                </Button>
              )}
            </div>,
          ];
        })}
        containerStyles={{ marginTop: "16px" }}
      />
    </div>
  );
};
