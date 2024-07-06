import React, { useState } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import {
  useBacktestById,
  useColumnQuery,
  useDatasetOhlcvCols,
  useDatasetQuery,
} from "../../../../clients/queries/queries";
import { TradesCandleStickChart } from "../../../../components/charts/TradesCandleStickChart";
import { Button, Spinner, Switch, useDisclosure } from "@chakra-ui/react";
import { WithLabel } from "../../../../components/form/WithLabel";
import { BUTTON_VARIANTS } from "../../../../theme";
import { ChakraPopover } from "../../../../components/chakra/popover";
import { OverflopTooltip } from "../../../../components/OverflowTooltip";
import { convertMillisToDateDict } from "../../../../utils/date";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

export const BacktestTradesPage = () => {
  const { backtestId, datasetName } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));
  const datasetQuery = useDatasetQuery(datasetName);
  const ohlcvColsData = useDatasetOhlcvCols(datasetName);
  const selectColumnPopover = useDisclosure();
  const [selectedColumnName, setSelectedColumnName] = useState("");
  const [showEnters, setShowEnters] = useState(true);
  const [showExits, setShowExits] = useState(true);
  const [showPriceTexts, setShowPriceTexts] = useState(false);
  const [showCustomColumn, setShowCustomColumn] = useState(true);
  const columnDetailedQuery = useColumnQuery(datasetName, selectedColumnName);

  if (!backtestQuery.data || !ohlcvColsData.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const getAltDataTicks = () => {
    const ticks = [];

    columnDetailedQuery.data?.res.column.rows.forEach((item, idx) => {
      const dateObj = convertMillisToDateDict(
        columnDetailedQuery.data.res.column.kline_open_time[idx]
      );
      ticks.push({ time: dateObj, value: item });
    });

    return ticks;
  };

  return (
    <div>
      <TradesCandleStickChart
        trades={backtestQuery.data.trades}
        ohlcvData={ohlcvColsData.data}
        isShortSellingTrades={backtestQuery.data.data.is_short_selling_strategy}
        hideTexts={!showPriceTexts}
        hideEnters={!showEnters}
        hideExits={!showExits}
        useAltData={columnDetailedQuery.data && showCustomColumn ? true : false}
        getAltDataTicks={
          columnDetailedQuery.data && showCustomColumn ? getAltDataTicks : null
        }
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
        <div>
          <ChakraPopover
            {...selectColumnPopover}
            setOpen={selectColumnPopover.onOpen}
            body={
              <div>
                {datasetQuery.data?.columns.map((item, idx: number) => {
                  return (
                    <div
                      key={idx}
                      className="link-default"
                      onClick={() => {
                        selectColumnPopover.onClose();
                        setSelectedColumnName(item);
                      }}
                    >
                      <OverflopTooltip text={item} containerId="COLUMN_MODAL">
                        <div>{item}</div>
                      </OverflopTooltip>
                    </div>
                  );
                })}
              </div>
            }
            headerText={"Visualize column"}
          >
            <Button variant={BUTTON_VARIANTS.nofill}>Add column</Button>
          </ChakraPopover>
        </div>

        {columnDetailedQuery.data && (
          <div>
            <WithLabel
              label={`${selectedColumnName}`}
              containerStyles={{ width: "max-content" }}
            >
              <Switch
                isChecked={showCustomColumn}
                onChange={() => setShowCustomColumn(!showCustomColumn)}
              />
            </WithLabel>
          </div>
        )}
      </div>
    </div>
  );
};
