import React from "react";
import { useBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { CreateBacktestDrawer } from "../../components/CreateNewBacktest";

export const BacktestUXManager = () => {
  const { createNewDrawer } = useBacktestContext();
  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "600px" }}
        {...createNewDrawer}
      >
        <CreateBacktestDrawer />
      </ChakraDrawer>
    </div>
  );
};
