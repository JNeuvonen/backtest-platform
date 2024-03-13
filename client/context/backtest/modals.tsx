import React from "react";
import { useBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { CreateBacktestDrawer } from "../../components/CreateNewBacktest";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";

export const BacktestUXManager = () => {
  const { createNewDrawer } = useBacktestContext();
  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "900px" }}
        {...createNewDrawer}
        footerContent={<FormSubmitBar />}
      >
        <CreateBacktestDrawer />
      </ChakraDrawer>
    </div>
  );
};
