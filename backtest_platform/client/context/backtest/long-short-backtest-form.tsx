import React, { useRef, useState } from "react";
import { useBacktestContext } from ".";
import { useDatasetQuery } from "../../clients/queries/queries";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { usePathParams } from "../../hooks/usePathParams";
import { Spinner, useDisclosure } from "@chakra-ui/react";
import { ShowColumnModal } from "./show-columns-modal";
import { BacktestFormControls } from "./backtest-form-controls";
import { FormikProps } from "formik";
import { DISK_KEYS, DiskManager } from "../../utils/disk";
import { useForceUpdate } from "../../hooks/useForceUpdate";

type PathParams = {
  datasetName: string;
};

const backtestDiskManager = new DiskManager(DISK_KEYS.backtest_form);

export const LongShortBacktestForm = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { longShortFormDrawer } = useBacktestContext();
  const columnsModal = useDisclosure();
  const { data: datasetQuery, refetch: refetchDataset } =
    useDatasetQuery(datasetName);
  const formikRef = useRef<FormikProps<any>>(null);
  const forceUpdate = useForceUpdate();

  if (!datasetQuery) return <Spinner />;

  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...longShortFormDrawer}
      >
        <div>
          <ShowColumnModal
            columns={datasetQuery.columns}
            datasetName={datasetName}
            columnsModal={columnsModal}
          />

          <BacktestFormControls
            columnsModal={columnsModal}
            formikRef={formikRef}
            backtestDiskManager={backtestDiskManager}
            forceUpdate={forceUpdate}
          />
        </div>
      </ChakraDrawer>
    </div>
  );
};
