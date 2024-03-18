import React from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useBacktestById } from "../../../../clients/queries/queries";
import { Heading, Spinner } from "@chakra-ui/react";
import { GenericAreaChart } from "../../../../components/charts/AreaChart";
import { FetchBacktestByIdRes } from "../../../../clients/queries/response-types";
import { ShareYAxisTwoLineChart } from "../../../../components/charts/ShareYAxisLineChart";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

interface PortfolioGrowthData {
  strategy: number;
  buy_and_hold: number;
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

export const DatasetBacktestPage = () => {
  const { datasetName, backtestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));

  if (!backtestQuery.data) return <Spinner />;

  return (
    <div>
      <Heading size={"lg"}>Backtest result</Heading>

      <ShareYAxisTwoLineChart
        data={getPortfolioGrowthData(backtestQuery.data)}
        xKey="kline_open_time"
        line1Key="strategy"
        line2Key="buy_and_hold"
        height={500}
        containerStyles={{ marginTop: "16px" }}
        showDots={false}
      />
    </div>
  );
};
