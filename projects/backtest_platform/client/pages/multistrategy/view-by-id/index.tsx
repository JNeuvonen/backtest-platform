import React, { useMemo } from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import { useBacktestById } from "../../../clients/queries/queries";
import {
  getDateRange,
  getPortfolioChartData,
} from "../../simulate/bulk/backtest-details-view";
import { FetchBacktestByIdRes } from "../../../clients/queries/response-types";
import { Heading, MenuButton, MenuItem, Spinner } from "@chakra-ui/react";
import { ChakraMenu } from "../../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa";
import { BacktestSummaryCard } from "../../simulate/dataset/backtest/SummaryCard";
import { ShareYAxisMultilineChart } from "../../../components/charts/ShareYAxisMultiline";
import { Line } from "recharts";
import { COLOR_BRAND_PRIMARY } from "../../../utils/colors";
import { saveBacktestReportMassSim } from "../../../clients/requests";

interface PathParams {
  backtestId: number;
}

export const MultiStrategyPageById = () => {
  const { backtestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));

  const portfolioTicks = useMemo(
    () => getPortfolioChartData(backtestQuery.data as FetchBacktestByIdRes),
    [backtestQuery.data]
  );

  if (!backtestQuery.data) return <Spinner />;

  const backtest = backtestQuery.data.data;

  const downloadDetailedSummary = async () => {
    await saveBacktestReportMassSim(Number(backtestId), "backtest_summary");
  };

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
          <MenuItem icon={<FaFileImport />} onClick={downloadDetailedSummary}>
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
    </div>
  );
};
