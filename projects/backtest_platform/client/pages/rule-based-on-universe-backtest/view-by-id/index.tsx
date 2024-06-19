import React, { useEffect, useMemo, useState } from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import { useBacktestById } from "../../../clients/queries/queries";
import { Heading, MenuButton, MenuItem, Spinner } from "@chakra-ui/react";
import { BacktestSummaryCard } from "../../simulate/dataset/backtest/SummaryCard";
import {
  getDateRange,
  getPortfolioChartData,
  getTradesData,
} from "../../simulate/bulk/backtest-details-view";
import { ShareYAxisMultilineChart } from "../../../components/charts/ShareYAxisMultiline";
import { Line } from "recharts";
import { COLOR_BRAND_PRIMARY } from "../../../utils/colors";
import { FetchBacktestByIdRes } from "../../../clients/queries/response-types";
import { TradingCriteriaCard } from "../../simulate/dataset/backtest/TradingCriteriaCard";
import { FaFileImport } from "react-icons/fa";
import { ChakraMenu } from "../../../components/chakra/Menu";
import { saveBacktestReport } from "../../../clients/requests";

interface PathParams {
  backtestId: number;
}

export const ViewRuleBasedMassBacktestPage = () => {
  const { backtestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));
  const [filterTradesRange, setFilterTradesRange] = useState([-500, 500]);

  const portfolioTicks = useMemo(
    () => getPortfolioChartData(backtestQuery.data as FetchBacktestByIdRes),
    [backtestQuery.data]
  );

  useEffect(() => {
    if (backtestQuery.data) {
      const trades = backtestQuery.data.trades.sort(
        (a, b) => a.percent_result - b.percent_result
      );
      const worstTrade = trades[0];
      const bestTrade = trades[trades.length - 1];

      if (worstTrade && bestTrade) {
        setFilterTradesRange([
          worstTrade.percent_result,
          bestTrade.percent_result,
        ]);
      }
    }
  }, [backtestQuery.data]);

  const downloadDetailedSummary = async () => {};

  if (!backtestQuery.data) return <Spinner />;

  const backtest = backtestQuery.data.data;

  if (backtest === undefined) {
    return null;
  }

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          width: "100%",
          justifyContent: "space-between",
        }}
      >
        <Heading size={"lg"}>
          Rule-based massbacktest {backtestQuery.data.data.name}
        </Heading>
      </div>

      <div>
        <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
          <MenuItem icon={<FaFileImport />} onClick={() => {}}>
            Detailed summary
          </MenuItem>
        </ChakraMenu>
      </div>

      <BacktestSummaryCard
        backtest={backtest}
        dateRange={getDateRange(portfolioTicks)}
      />
      <Heading size={"md"} marginTop={"16px"}>
        Backtest balance growth
      </Heading>

      <div>
        <div style={{ marginTop: "16px" }}>
          <ShareYAxisMultilineChart
            height={500}
            data={portfolioTicks}
            xAxisKey={"kline_open_time"}
            xAxisTickFormatter={(tick: number) => {
              return new Date(tick).toLocaleDateString("default", {
                year: "numeric",
                month: "short",
              });
            }}
          >
            <Line
              type="monotone"
              dataKey={"strategy"}
              stroke={COLOR_BRAND_PRIMARY}
              dot={false}
            />
          </ShareYAxisMultilineChart>
        </div>
      </div>

      <TradingCriteriaCard backtestQuery={backtestQuery} />
    </div>
  );
};
