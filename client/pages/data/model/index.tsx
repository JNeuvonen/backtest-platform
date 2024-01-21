import React from "react";
// import usePath from "../../../hooks/usePath";
import { ChakraModal } from "../../../components/chakra/modal";
import { useModal } from "../../../hooks/useOpen";
import { Button, Spinner } from "@chakra-ui/react";
import { DatasetModelCreatePage } from "./create";
import { useDatasetModelsQuery } from "../../../clients/queries/queries";
import { usePathParams } from "../../../hooks/usePathParams";
import { SmallTable } from "../../../components/tables/Small";
import { formatValidationSplit } from "../../../utils/content";
import { useNavigate } from "react-router-dom";

// const TAB_LABELS = ["Available", "Create", "Train"];

const MODEL_COLUMNS = ["Name", "Target Column", "Validation Split"];
export const DatasetModelIndex = () => {
  // const { path } = usePath();

  const createModelModal = useModal();
  const { datasetName } = usePathParams<{ datasetName: string }>();
  const { data } = useDatasetModelsQuery(datasetName);
  const navigate = useNavigate();

  if (!data?.res) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const modelsArr = data.res.data;

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
      <SmallTable
        containerStyles={{ marginTop: "16px" }}
        columns={MODEL_COLUMNS}
        rows={modelsArr.map((item) => {
          return [
            item.name,
            item.target_col,
            formatValidationSplit(item.validation_split),
          ];
        })}
        rowOnClickFunc={() => navigate("/")}
      />
    </div>
  );
};
