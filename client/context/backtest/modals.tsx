import React, { useState } from "react";
import { useBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { CreateBacktestDrawer } from "../../components/CreateNewBacktest";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";
import {
  ENTER_TRADE_DEFAULT,
  EXIT_LONG_TRADE_DEFAULT,
  EXIT_SHORT_TRADE_DEFAULT,
  EXIT_TRADE_DEFAULT,
} from "../../utils/code";
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

  const { createNewDrawer, datasetBacktestsQuery, forceUpdate } =
    useBacktestContext();

  const [openLongTradeCode, setOpenLongTradeCode] = useState(
    ENTER_TRADE_DEFAULT()
  );
  const [openShortTradeCode, setOpenShortTradeCode] =
    useState(EXIT_TRADE_DEFAULT());

  const [closeLongTradeCode, setCloseLongTradeCode] = useState(
    EXIT_LONG_TRADE_DEFAULT()
  );
  const [closeShortTradeCode, setCloseShortTradeCode] = useState(
    EXIT_SHORT_TRADE_DEFAULT()
  );
  const [backtestName, setBacktestName] = useState("");
  const [useShorts, setUseShorts] = useState(false);

  const [useTimeBasedClose, setUseTimeBasedClose] = useState(false);
  const [klinesUntilClose, setKlinesUntilClose] = useState<null | number>(null);

  const toast = useToast();

  const submitNewBacktest = async () => {
    if (!dataset) return;

    const res = await createManualBacktest({
      open_long_trade_cond: openLongTradeCode,
      close_long_trade_cond: closeLongTradeCode,
      open_short_trade_cond: openShortTradeCode,
      close_short_trade_cond: closeShortTradeCode,
      use_short_selling: useShorts,
      dataset_id: dataset.id,
      name: backtestName,
      use_time_based_close: useTimeBasedClose,
      klines_until_close: klinesUntilClose,
    });

    if (res.status === 200) {
      toast({
        title: "Finished backtest",
        description: `Result: ${
          res.res.data.end_balance - res.res.data.start_balance
        }`,
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      createNewDrawer.onClose();
      datasetBacktestsQuery.refetch();
      forceUpdate();
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
          klinesUntilClose={klinesUntilClose}
          setKlinesUntilClose={setKlinesUntilClose}
          openLongTradeCode={openLongTradeCode}
          setOpenLongTradeCode={setOpenLongTradeCode}
          openShortTradeCode={openShortTradeCode}
          closeLongTradeCode={closeLongTradeCode}
          setCloseLongTradeCode={setCloseLongTradeCode}
          closeShortTradeCode={closeShortTradeCode}
          setCloseShortTradeCode={setCloseShortTradeCode}
          setOpenShortTradeCode={setOpenShortTradeCode}
          useShorts={useShorts}
          setUseShorts={setUseShorts}
          backtestName={backtestName}
          setBacktestName={setBacktestName}
          useTimeBasedClose={useTimeBasedClose}
          setUseTimeBasedClose={setUseTimeBasedClose}
        />
      </ChakraDrawer>
    </div>
  );
};
