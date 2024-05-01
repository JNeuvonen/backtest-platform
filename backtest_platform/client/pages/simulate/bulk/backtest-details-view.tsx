import React, { useState } from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import { useBacktestById } from "../../../clients/queries/queries";
import { Heading, Spinner, Switch } from "@chakra-ui/react";
import { BacktestSummaryCard } from "../dataset/backtest/SummaryCard";
import { ShareYAxisTwoLineChart } from "../../../components/charts/ShareYAxisLineChart";
import { FetchBacktestByIdRes } from "../../../clients/queries/response-types";
import { ShareYAxisMultilineChart } from "../../../components/charts/ShareYAxisMultiline";
import { Line } from "recharts";
import {
  COLOR_BRAND_PRIMARY,
  COLOR_BRAND_SECONDARY,
} from "../../../utils/colors";
import { WithLabel } from "../../../components/form/WithLabel";

interface PathParams {
  massPairTradeBacktestId: number;
}

interface ChartTick {
  strategy: number;
  benchmark: number;
  kline_open_time: number;
}

const getPortfolioChartData = (backtestData: FetchBacktestByIdRes) => {
  const ret = [] as ChartTick[];

  const portfolioData = backtestData.balance_history;

  const increment = Math.max(Math.floor(portfolioData.length / 1000), 1);

  for (let i = 0; i < portfolioData.length; i += increment) {
    const item = portfolioData[i];
    ret.push({
      strategy: item.portfolio_worth,
      benchmark: item.benchmark_price,
      kline_open_time: item.kline_open_time * 1000,
    });
  }
  return ret;
};

const getDateRange = (portfolioTicks: ChartTick[]): string => {
  const firstItem = portfolioTicks[0].kline_open_time;
  const lastItem = portfolioTicks[portfolioTicks.length - 1].kline_open_time;

  const firstDate = new Date(firstItem).toLocaleDateString("default", {
    year: "numeric",
    month: "short",
  });
  const lastDate = new Date(lastItem).toLocaleDateString("default", {
    year: "numeric",
    month: "short",
  });

  return `${firstDate} - ${lastDate}`;
};

export const LongShortBacktestsDetailsView = () => {
  const { massPairTradeBacktestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(massPairTradeBacktestId));
  const [hideBenchmark, setHideBenchmark] = useState(false);

  if (backtestQuery.isLoading || !backtestQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const backtest = backtestQuery.data.data;
  const portfolioTicks = getPortfolioChartData(backtestQuery.data);

  return (
    <div>
      <div>
        <Heading size={"lg"}>Pair-trade backtest {backtest.name}</Heading>
      </div>
      <BacktestSummaryCard
        backtest={backtest}
        dateRange={getDateRange(portfolioTicks)}
      />

      <div>
        <WithLabel label={"Hide benchmark"}>
          <Switch
            isChecked={hideBenchmark}
            onChange={() => setHideBenchmark(!hideBenchmark)}
          />
        </WithLabel>
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
            {!hideBenchmark && (
              <Line
                type="monotone"
                dataKey={"benchmark"}
                stroke={COLOR_BRAND_SECONDARY}
                dot={false}
              />
            )}
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
