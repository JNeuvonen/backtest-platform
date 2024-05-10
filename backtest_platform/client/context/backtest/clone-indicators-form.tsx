import React from "react";
import { useBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { CloneIndicatorsDrawer } from "../../components/CloneIndicatorsForm";

export const BacktestCloneIndicators = () => {
  const { cloneIndicatorsDrawer, datasetName } = useBacktestContext();
  return (
    <div>
      <ChakraDrawer
        title="Clone indicators to another dataset"
        drawerContentStyles={{ maxWidth: "50%" }}
        {...cloneIndicatorsDrawer}
      >
        <CloneIndicatorsDrawer
          datasetName={datasetName}
          cancelCallback={cloneIndicatorsDrawer.onClose}
          submitCallback={() => {
            cloneIndicatorsDrawer.onClose();
          }}
        />
      </ChakraDrawer>
    </div>
  );
};
