import React, { CSSProperties } from "react";
import { Trade } from "../clients/queries/response-types";
import { findTradesByYear } from "../utils/backtest";
import { GenericTable } from "./tables/GenericTable";
import { Button, Heading } from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";
import { getNumberDisplayColor, roundNumberFloor } from "common_js";
import { COLOR_CONTENT_PRIMARY } from "../utils/colors";

interface Props {
  trades: Trade[];
  onTradeClickCallback: (trade: Trade) => void;
  style?: CSSProperties;
}

const COLUMNS = [
  "View on chart",
  "Open price",
  "Close Price",
  "Net result",
  "% Result",
];

export const TradesByYearTable = ({
  trades,
  onTradeClickCallback,
  style,
}: Props) => {
  const createRows = (trades: Trade[]) => {
    const sortedTrades = trades.sort((a, b) => b.open_time - a.open_time);
    const rows: any[] = [];

    sortedTrades.forEach((item) => {
      rows.push([
        <Button
          variant={BUTTON_VARIANTS.nofill}
          onClick={() => onTradeClickCallback(item)}
        >
          {new Date(item.open_time).toDateString()}
        </Button>,
        item.open_price,
        item.close_price,
        <div
          style={{
            color: getNumberDisplayColor(
              item.net_result,
              COLOR_CONTENT_PRIMARY
            ),
          }}
        >
          {roundNumberFloor(item.net_result, 0)}
        </div>,
        <div
          style={{
            color: getNumberDisplayColor(
              item.net_result,
              COLOR_CONTENT_PRIMARY
            ),
          }}
        >
          {roundNumberFloor(item.percent_result, 2)}
        </div>,
      ]);
    });

    return rows;
  };

  const renderTables = () => {
    const years: Set<number> = new Set();

    trades.forEach((item) => {
      const date = new Date(item.open_time);

      years.add(date.getUTCFullYear());
    });

    const yearsArr: number[] = Array.from(years).sort((a, b) => b - a);

    return (
      <div>
        {yearsArr.map((year) => {
          const tradesByYear = findTradesByYear(trades, year);
          return (
            <div style={{ marginTop: "16px" }}>
              <Heading size={"s"}>
                {year} - trades {tradesByYear.length}
              </Heading>
              <div style={{ marginTop: "8px" }}>
                <GenericTable
                  columns={COLUMNS}
                  rows={createRows(tradesByYear)}
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  };
  return <div style={style}>{renderTables()}</div>;
};
