import React from "react";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { useMLBasedBacktestContext } from ".";

export const CreateNewMLBasedBacktestDrawer = () => {
  const { createNewDrawer } = useMLBasedBacktestContext();
  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...createNewDrawer}
      >
        <div>Hello world</div>
      </ChakraDrawer>
    </div>
  );
};
