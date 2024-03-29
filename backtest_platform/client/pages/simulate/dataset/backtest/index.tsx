import React, { useState } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useBacktestById } from "../../../../clients/queries/queries";
import {
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
} from "@chakra-ui/react";
import { GenericAreaChart } from "../../../../components/charts/AreaChart";
import {
  BacktestBalance,
  FetchBacktestByIdRes,
  Trade,
} from "../../../../clients/queries/response-types";
import { ShareYAxisTwoLineChart } from "../../../../components/charts/ShareYAxisLineChart";
import { GenericBarChart } from "../../../../components/charts/BarChart";
import { ChakraSlider } from "../../../../components/chakra/Slider";
import { ChakraCard } from "../../../../components/chakra/Card";
import { COLOR_CONTENT_PRIMARY } from "../../../../utils/colors";
import { roundNumberDropRemaining } from "../../../../utils/number";

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

const isTradeExitTick = (trades: Trade[], kline_open_time: number) => {
  const found = trades.filter((trade) =>
    trade && trade.close_time ? trade.close_time === kline_open_time : false
  );
  return found.length > 0;
};

const isTradeEnterTick = (trades: Trade[], kline_open_time: number) => {
  const found = trades.filter((trade) =>
    trade && trade.open_time ? trade.open_time === kline_open_time : false
  );
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
  const { backtestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));

  const [tradeFilterPerc, setTradeFilterPerc] = useState(0);

  if (!backtestQuery.data) return <Spinner />;

  const backtest = backtestQuery.data.data;

  return (
    <div>
      <Heading size={"lg"}>Backtest {backtestQuery.data.data.name}</Heading>

      <div style={{ marginTop: "16px" }}>
        <ChakraCard heading={<Heading size="md">Summary</Heading>}>
          <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Risk adjusted CAGR</StatLabel>
                <StatNumber>
                  {backtest.risk_adjusted_return
                    ? String(
                        roundNumberDropRemaining(
                          backtest.risk_adjusted_return * 100,
                          2
                        )
                      ) + "%"
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Actual CAGR</StatLabel>
                <StatNumber>
                  {backtest.cagr
                    ? String(roundNumberDropRemaining(backtest.cagr * 100, 2)) +
                      "%"
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Buy and hold CAGR</StatLabel>
                <StatNumber>
                  {backtest.buy_and_hold_cagr
                    ? String(
                        roundNumberDropRemaining(
                          backtest.buy_and_hold_cagr * 100,
                          2
                        )
                      ) + "%"
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Time exposure</StatLabel>
                <StatNumber>
                  {backtest.market_exposure_time
                    ? String(
                        roundNumberDropRemaining(
                          backtest.market_exposure_time * 100,
                          2
                        )
                      ) + "%"
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Max drawdown</StatLabel>
                <StatNumber>
                  {backtest.max_drawdown_perc
                    ? String(
                        roundNumberDropRemaining(backtest.max_drawdown_perc, 2)
                      ) + "%"
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>

            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Profit factor</StatLabel>
                <StatNumber>
                  {backtest.profit_factor
                    ? String(
                        roundNumberDropRemaining(backtest.profit_factor, 2)
                      )
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>
          </div>
        </ChakraCard>
      </div>

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
