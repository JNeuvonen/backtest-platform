import React, { useState } from "react";
import { useBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { CreateBacktestDrawer } from "../../components/CreateNewBacktest";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";
import { ENTER_TRADE_DEFAULT, EXIT_TRADE_DEFAULT } from "../../utils/code";
import { createManualBacktest } from "../../clients/requests";
import { usePathParams } from "../../hooks/usePathParams";
import { useDatasetQuery } from "../../clients/queries/queries";
import { useToast } from "@chakra-ui/react";

type PathParams = {
  datasetName: string;
};

export const BacktestUXManager = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { data: dataset } = useDatasetQuery(datasetName);

  const { createNewDrawer } = useBacktestContext();
  const [enterTradeCode, setEnterTradeCode] = useState(ENTER_TRADE_DEFAULT());
  const [exitTradeCode, setExitTradeCode] = useState(EXIT_TRADE_DEFAULT());
  const [doNotShort, setDoNotShort] = useState(true);

  const toast = useToast();

  const submitNewBacktest = async () => {
    if (!dataset) return;

    const res = await createManualBacktest({
      enter_trade_cond: enterTradeCode,
      exit_trade_cond: exitTradeCode,
      use_short_selling: !doNotShort,
      dataset_id: dataset.id,
    });

    if (res.status === 200) {
      toast({
        title: "Created backtest",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      createNewDrawer.onClose();
    }
  };

  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "900px" }}
        {...createNewDrawer}
        footerContent={
          <FormSubmitBar
            submitCallback={submitNewBacktest}
            cancelCallback={createNewDrawer.onClose}
          />
        }
      >
        <CreateBacktestDrawer
          enterTradeCode={enterTradeCode}
          setEnterTradeCode={setEnterTradeCode}
          exitTradeCode={exitTradeCode}
          setExitTradeCode={setExitTradeCode}
          doNotShort={doNotShort}
          setDoNotShort={setDoNotShort}
        />
      </ChakraDrawer>
    </div>
  );
};
