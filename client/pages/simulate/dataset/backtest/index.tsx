import React, { useState } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useBacktestById } from "../../../../clients/queries/queries";
import { Heading, Spinner } from "@chakra-ui/react";
import { GenericAreaChart } from "../../../../components/charts/AreaChart";
import {
  FetchBacktestByIdRes,
  Trade,
} from "../../../../clients/queries/response-types";
import { ShareYAxisTwoLineChart } from "../../../../components/charts/ShareYAxisLineChart";
import { GenericBarChart } from "../../../../components/charts/BarChart";
import { ChakraSlider } from "../../../../components/chakra/Slider";
import { ChakraCard } from "../../../../components/chakra/Card";
import { COLOR_CONTENT_PRIMARY } from "../../../../utils/colors";

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

  for (let i = 0; i < portfolioData.length; i += increment) {
    const item = portfolioData[i];

    const isEnterTick = isTradeEnterTick(trades, item["kline_open_time"]);
    const isExitTick = isTradeExitTick(trades, item["kline_open_time"]);

    ret.push({
      strategy: item["portfolio_worth"],
      kline_open_time: item["kline_open_time"],
      buy_and_hold: item["buy_and_hold_worth"],
      enter_trade: isEnterTick ? item["buy_and_hold_worth"] : null,
      exit_trade: isExitTick ? item["buy_and_hold_worth"] : null,
    });
  }

  return ret;
};

const isTradeExitTick = (trades: Trade[], kline_open_time: number) => {
  const found = trades.filter((trade) => trade.close_time === kline_open_time);
  return found.length > 0;
};

const isTradeEnterTick = (trades: Trade[], kline_open_time: number) => {
  const found = trades.filter((trade) => trade.open_time === kline_open_time);
  return found.length > 0;
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

      <div style={{ marginTop: "16px" }}>
        <ChakraCard heading={<Heading size="md">Trading criteria</Heading>}>
          <pre style={{ color: COLOR_CONTENT_PRIMARY }}>
            {backtestQuery.data.data.open_long_trade_cond}
          </pre>
          <pre style={{ marginTop: "8px", color: COLOR_CONTENT_PRIMARY }}>
            {backtestQuery.data.data.close_long_trade_cond}
          </pre>

          {backtestQuery.data.data.use_short_selling && (
            <>
              <pre style={{ marginTop: "8px", color: COLOR_CONTENT_PRIMARY }}>
                {backtestQuery.data.data.open_short_trade_cond}
              </pre>
              <pre style={{ marginTop: "8px", color: COLOR_CONTENT_PRIMARY }}>
                {backtestQuery.data.data.close_short_trade_cond}
              </pre>
            </>
          )}
        </ChakraCard>
      </div>

      <div style={{ marginTop: "16px" }}>
        <ChakraCard heading={<Heading size="md">Strategy</Heading>}>
          <pre style={{ color: COLOR_CONTENT_PRIMARY }}>
            Use short selling:{" "}
            {backtestQuery.data.data.use_short_selling ? "True" : "False"}
          </pre>
          <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
            Use time based close:{" "}
            {backtestQuery.data.data.use_time_based_close
              ? `True, ${backtestQuery.data.data.klines_until_close} candles`
              : "False"}
          </pre>
          <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
            Use stop loss based close:{" "}
            {backtestQuery.data.data.use_stop_loss_based_close
              ? `True, ${backtestQuery.data.data.stop_loss_threshold_perc}%`
              : "False"}
          </pre>
          <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
            Use profit (%) based close:{" "}
            {backtestQuery.data.data.use_profit_based_close
              ? `True, ${backtestQuery.data.data.take_profit_threshold_perc}%`
              : "False"}
          </pre>
        </ChakraCard>
      </div>
    </div>
  );
};
