import React, { useState } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import {
  useBacktestById,
  useDatasetOhlcvCols,
} from "../../../../clients/queries/queries";
import { TradesCandleStickChart } from "../../../../components/charts/TradesCandleStickChart";
import { Spinner, Switch } from "@chakra-ui/react";
import { WithLabel } from "../../../../components/form/WithLabel";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

export const BacktestTradesPage = () => {
  const { backtestId, datasetName } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));
  const ohlcvColsData = useDatasetOhlcvCols(datasetName);
  const [showEnters, setShowEnters] = useState(true);
  const [showExits, setShowExits] = useState(true);
  const [showPriceTexts, setShowPriceTexts] = useState(false);

  if (!backtestQuery.data || !ohlcvColsData.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  return (
    <div>
      <TradesCandleStickChart
        trades={backtestQuery.data.trades}
        ohlcvData={ohlcvColsData.data}
        isShortSellingTrades={backtestQuery.data.data.is_short_selling_strategy}
        hideTexts={!showPriceTexts}
        hideEnters={!showEnters}
        hideExits={!showExits}
      />
      <div
        style={{
          marginTop: "16px",
          display: "flex",
          alignItems: "center",
          gap: "16px",
        }}
      >
        <div>
          <WithLabel
            label={"Show price texts"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={showPriceTexts}
              onChange={() => setShowPriceTexts(!showPriceTexts)}
            />
          </WithLabel>
        </div>
        <div>
          <WithLabel
            label={"Show enters"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={showEnters}
              onChange={() => setShowEnters(!showEnters)}
            />
          </WithLabel>
        </div>
        <div>
          <WithLabel
            label={"Show exits"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={showExits}
              onChange={() => setShowExits(!showExits)}
            />
          </WithLabel>
        </div>
      </div>
    </div>
  );
};
