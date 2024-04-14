import React, { useState } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useBacktestById } from "../../../../clients/queries/queries";
import { Button, Heading, Spinner, useDisclosure } from "@chakra-ui/react";
import {
  BacktestBalance,
  FetchBacktestByIdRes,
} from "../../../../clients/queries/response-types";
import { ShareYAxisTwoLineChart } from "../../../../components/charts/ShareYAxisLineChart";
import { GenericBarChart } from "../../../../components/charts/BarChart";
import { ChakraSlider } from "../../../../components/chakra/Slider";
import { TradingCriteriaCard } from "./TradingCriteriaCard";
import { BacktestSummaryCard } from "./SummaryCard";
import { GrDeploy } from "react-icons/gr";
import { useBacktestContext } from "../../../../context/backtest";
import { DeployStrategyForm } from "./DeployStrategyForm";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

interface PortfolioGrowthData {
  strategy: number;
  buy_and_hold: number;
  kline_open_time: number;
  enter_trade: null | boolean;
  exit_trade: null | boolean;
}

interface TradesBarChartData {
  perc_result: number;
  kline_open_time: number;
}

const getPortfolioGrowthData = (backtestData: FetchBacktestByIdRes) => {
  const ret = [] as PortfolioGrowthData[];

  const portfolioData = backtestData.data.data;
  const increment = Math.max(Math.floor(portfolioData.length / 3650), 1);

  const trades = backtestData.trades;

  const setOfUsedKlines = new Set();

  for (let i = 0; i < trades.length; i++) {
    const tickEnter = findTickBasedOnOpenTime(
      portfolioData,
      trades[i].open_time
    );
    const tickClose = findTickBasedOnOpenTime(
      portfolioData,
      trades[i].close_time
    );

    if (!tickEnter || !tickClose) {
      continue;
    }

    ret.push({
      strategy: tickEnter["portfolio_worth"],
      kline_open_time: tickEnter["kline_open_time"],
      buy_and_hold: tickEnter["buy_and_hold_worth"],
      enter_trade: tickEnter["buy_and_hold_worth"],
      exit_trade: null,
    });

    ret.push({
      strategy: tickClose["portfolio_worth"],
      kline_open_time: tickClose["kline_open_time"],
      buy_and_hold: tickClose["buy_and_hold_worth"],
      enter_trade: null,
      exit_trade: tickClose["buy_and_hold_worth"],
    });
    setOfUsedKlines.add(tickEnter["kline_open_time"]);
    setOfUsedKlines.add(tickClose["kline_open_time"]);
  }

  for (let i = 0; i < portfolioData.length; i += increment) {
    const item = portfolioData[i];
    if (setOfUsedKlines.has(item["kline_open_time"])) {
      continue;
    }
    ret.push({
      strategy: item["portfolio_worth"],
      kline_open_time: item["kline_open_time"],
      buy_and_hold: item["buy_and_hold_worth"],
      enter_trade: null,
      exit_trade: null,
    });
  }

  ret.sort((a, b) => a.kline_open_time - b.kline_open_time);

  return ret;
};

const findTickBasedOnOpenTime = (
  ticks: BacktestBalance[],
  kline_open_time: number
) => {
  const tick = ticks.filter((item) => item.kline_open_time === kline_open_time);
  return tick.length > 0 ? tick[0] : null;
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
  const { backtestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));
  const deployStrategyDrawer = useDisclosure();

  const [tradeFilterPerc, setTradeFilterPerc] = useState(0);

  if (!backtestQuery.data) return <Spinner />;

  const backtest = backtestQuery.data.data;

  if (backtest === undefined) {
    return null;
  }

  return (
    <>
      <DeployStrategyForm deployStrategyDrawer={deployStrategyDrawer} />
      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            width: "100%",
            justifyContent: "space-between",
          }}
        >
          <Heading size={"lg"}>Backtest {backtestQuery.data.data.name}</Heading>
          <Button leftIcon={<GrDeploy />} onClick={deployStrategyDrawer.onOpen}>
            Deploy
          </Button>
        </div>
        <BacktestSummaryCard backtest={backtest} />
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
          displayTradeEntersAndExits={true}
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
        <TradingCriteriaCard backtestQuery={backtestQuery} />
      </div>
    </>
  );
};
