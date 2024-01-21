import React from "react";
// import usePath from "../../../hooks/usePath";
import { ChakraModal } from "../../../components/chakra/modal";
import { useModal } from "../../../hooks/useOpen";
import { Button, Spinner } from "@chakra-ui/react";
import { DatasetModelCreatePage } from "./create";
import { useDatasetModelsQuery } from "../../../clients/queries/queries";
import { usePathParams } from "../../../hooks/usePathParams";
import { RowItem, SmallTable } from "../../../components/tables/Small";
import { formatValidationSplit } from "../../../utils/content";
import { Outlet, useNavigate } from "react-router-dom";
import { getModelInfoPath } from "../../../utils/navigate";
import { PATHS, PATH_KEYS } from "../../../utils/constants";
import usePath from "../../../hooks/usePath";

// const TAB_LABELS = ["Available", "Create", "Train"];

const MODEL_COLUMNS = ["Name", "Target Column", "Validation Split"];
const ROOT_PATH = PATHS.data.model.index;

export const DatasetModelIndex = () => {
  const { path } = usePath();

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

  const pathIsRoot = () => {
    return path === ROOT_PATH.replace(PATH_KEYS.dataset, datasetName);
  };

  const modelsArr = data.res.data;

  return (
    <div>
      {pathIsRoot() && (
        <>
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
            rowOnClickFunc={(item: RowItem) => {
              navigate(getModelInfoPath(datasetName, item[0] as string));
            }}
          />
        </>
      )}

      <Outlet />
    </div>
  );
};
