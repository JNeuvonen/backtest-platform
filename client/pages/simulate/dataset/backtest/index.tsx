import React, { useState } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useBacktestById } from "../../../../clients/queries/queries";
import { Heading, Spinner } from "@chakra-ui/react";
import { GenericAreaChart } from "../../../../components/charts/AreaChart";
import { FetchBacktestByIdRes } from "../../../../clients/queries/response-types";
import { ShareYAxisTwoLineChart } from "../../../../components/charts/ShareYAxisLineChart";
import { GenericBarChart } from "../../../../components/charts/BarChart";
import { ChakraSlider } from "../../../../components/chakra/Slider";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

interface PortfolioGrowthData {
  strategy: number;
  buy_and_hold: number;
  kline_open_time: number;
}

interface TradesBarChartData {
  perc_result: number;
  kline_open_time: number;
}

const getPortfolioGrowthData = (backtestData: FetchBacktestByIdRes) => {
  const ret = [] as PortfolioGrowthData[];

  const portfolioData = backtestData.data.data;
  const increment = Math.max(Math.floor(portfolioData.length / 500), 1);

  for (let i = 0; i < portfolioData.length; i += increment) {
    const item = portfolioData[i];

    ret.push({
      strategy: item["portfolio_worth"],
      kline_open_time: item["kline_open_time"],
      buy_and_hold: item["buy_and_hold_worth"],
    });
  }

  return ret;
};

const getTradesData = (
  backtestData: FetchBacktestByIdRes,
  percFilter: number
) => {
  const ret = [] as TradesBarChartData[];

  const trades = backtestData.trades.sort(
    (a, b) => a.percent_result - b.percent_result
  );

  for (let i = 0; i < trades.length; i++) {
    if (Math.abs(trades[i].percent_result) < percFilter) {
      continue;
    }
    ret.push({
      perc_result: trades[i].percent_result,
      kline_open_time: trades[i].open_time,
    });
  }

  return ret;
};

export const DatasetBacktestPage = () => {
  const { datasetName, backtestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));

  const [tradeFilterPerc, setTradeFilterPerc] = useState(0);

  if (!backtestQuery.data) return <Spinner />;

  return (
    <div>
      <Heading size={"lg"}>Backtest {backtestQuery.data.data.name}</Heading>

      <Heading size={"md"} marginTop={"16px"}>
        Backtest balance growth
      </Heading>

      <ShareYAxisTwoLineChart
        data={getPortfolioGrowthData(backtestQuery.data)}
        xKey="kline_open_time"
        line1Key="strategy"
        line2Key="buy_and_hold"
        height={500}
        containerStyles={{ marginTop: "16px" }}
        showDots={false}
      />

      <Heading size={"md"}>Trade results</Heading>
      <ChakraSlider
        label={`Filter trades: ${tradeFilterPerc}%`}
        containerStyles={{ maxWidth: "300px", marginTop: "16px" }}
        min={0}
        max={50}
        onChange={setTradeFilterPerc}
        defaultValue={0}
        value={tradeFilterPerc}
      />
      <GenericBarChart
        data={getTradesData(backtestQuery.data, tradeFilterPerc)}
        yAxisKey="perc_result"
        xAxisKey="kline_open_time"
        containerStyles={{ marginTop: "16px" }}
      />
    </div>
  );
};
