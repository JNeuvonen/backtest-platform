import React, { useEffect } from "react";
import { ChakraModal } from "../../../components/chakra/modal";
import { useModal } from "../../../hooks/useOpen";
import { Button, Spinner } from "@chakra-ui/react";
import { DatasetModelCreatePage } from "./create";
import { useDatasetModelsQuery } from "../../../clients/queries/queries";
import { usePathParams } from "../../../hooks/usePathParams";
import { RowItem, SmallTable } from "../../../components/tables/Small";
import { useNavigate } from "react-router-dom";
import { getModelInfoPath } from "../../../utils/navigate";
import {
  LAYOUT,
  PATHS,
  PATH_KEYS,
  formatValidationSplit,
} from "../../../utils/constants";
import usePath from "../../../hooks/usePath";
import { ModelInfoPage } from "./info";
import useQueryParams from "../../../hooks/useQueryParams";
import { ChakraTabs } from "../../../components/layout/Tabs";
import { ModelTrainPage } from "./Train";
import { displayValidationSplit } from "../../../utils/content";
import { useAppContext } from "../../../context/App";

const TAB_LABELS = ["Info", "Train"];
const TABS = [<ModelInfoPage key={"1"} />, <ModelTrainPage key={"2"} />];

interface QueryParams {
  defaultTab: string | undefined;
}

const MODEL_COLUMNS = ["Name", "Target Column", "Validation Split"];
const ROOT_PATH = PATHS.data.model.index;

export const DatasetModelIndex = () => {
  const { path } = usePath();
  const { defaultTab } = useQueryParams<QueryParams>();
  const { setInnerSideNavWidth } = useAppContext();

  const createModelModal = useModal();
  const { datasetName } = usePathParams<{ datasetName: string }>();
  const { data, refetch } = useDatasetModelsQuery(datasetName);
  const navigate = useNavigate();

  useEffect(() => {
    setInnerSideNavWidth(LAYOUT.inner_side_nav_width_px);
    return () => setInnerSideNavWidth(0);
  }, []);

  if (!data || !data?.res || data.status !== 200) {
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
            <DatasetModelCreatePage
              cancelCallback={createModelModal.onClose}
              submitCallback={() => {
                createModelModal.onClose();
                refetch();
              }}
            />
          </ChakraModal>

          <Button onClick={createModelModal.onOpen}>Create</Button>

          <SmallTable
            containerStyles={{ marginTop: "16px" }}
            columns={MODEL_COLUMNS}
            rows={modelsArr.map((item) => {
              return [
                item.model_name,
                item.target_col,
                displayValidationSplit(
                  formatValidationSplit(item.validation_split)
                ),
              ];
            })}
            rowOnClickFunc={(item: RowItem) => {
              navigate(getModelInfoPath(datasetName, item[0] as string));
            }}
          />
        </>
      )}

      {!pathIsRoot() && (
        <ChakraTabs
          labels={TAB_LABELS}
          tabs={TABS}
          defaultTab={defaultTab ? Number(defaultTab) : 0}
        />
      )}
    </div>
  );
};
