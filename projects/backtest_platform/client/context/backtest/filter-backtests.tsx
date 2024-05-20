import React from "react";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { useBacktestContext } from ".";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";

export const FilterBacktestDrawer = () => {
  const { filterDrawer } = useBacktestContext();
  return (
    <div>
      <ChakraDrawer
        title="Filter backtests"
        drawerContentStyles={{ maxWidth: "40%" }}
        {...filterDrawer}
        footerContent={<FormSubmitBar submitText="Apply" />}
      >
        <div>Filter drawer</div>
      </ChakraDrawer>
    </div>
  );
};
