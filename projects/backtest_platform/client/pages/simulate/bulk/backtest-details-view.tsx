import React, { useEffect, useMemo, useState } from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useBacktestById,
  useMassbacktestSymbols,
  useMassbacktestTransformations,
} from "../../../clients/queries/queries";
import {
  Button,
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
  Switch,
  useDisclosure,
} from "@chakra-ui/react";
import { BacktestSummaryCard } from "../dataset/backtest/SummaryCard";
import {
  BacktestObject,
  FetchBacktestByIdRes,
} from "../../../clients/queries/response-types";
import { ShareYAxisMultilineChart } from "../../../components/charts/ShareYAxisMultiline";
import { Line } from "recharts";
import {
  COLOR_BRAND_PRIMARY,
  COLOR_BRAND_SECONDARY,
  COLOR_CONTENT_PRIMARY,
} from "../../../utils/colors";
import { WithLabel } from "../../../components/form/WithLabel";
import { GenericBarChart } from "../../../components/charts/BarChart";
import { TradesBarChartData } from "../dataset/backtest";
import { roundNumberDropRemaining, safeDivide } from "../../../utils/number";
import { GenericRangeSlider } from "../../../components/chakra/RangeSlider";
import { ChakraCard } from "../../../components/chakra/Card";
import { GrDeploy } from "react-icons/gr";
import { ChakraDrawer } from "../../../components/chakra/Drawer";
import { DeployLongShortStrategyForm } from "./deployform";

interface PathParams {
  massPairTradeBacktestId: number;
}

interface ChartTick {
  strategy: number;
  benchmark: number;
  kline_open_time: number;
}

export const getPortfolioChartData = (backtestData: FetchBacktestByIdRes) => {
  if (backtestData === undefined) return [];

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

export const getTradesData = (
  backtestData: FetchBacktestByIdRes,
  filterTradesRange: number[],
  pairTrades: boolean = true
) => {
  if (backtestData === undefined) return {};

  const ret = [] as TradesBarChartData[];
  let cumulativeResults = 0;
  let numWinningTrades = 0;
  let numLosingTrades = 0;

  const trades = backtestData[pairTrades ? "pair_trades" : "trades"].sort(
    (a, b) => a.percent_result - b.percent_result
  );

  for (let i = 0; i < trades.length; ++i) {
    if (
      trades[i].percent_result < filterTradesRange[0] ||
      trades[i].percent_result > filterTradesRange[1]
    ) {
      continue;
    }
    ret.push({
      perc_result: trades[i].percent_result,
      kline_open_time: trades[i].open_time,
    });
    cumulativeResults += trades[i].percent_result;

    if (trades[i].percent_result < 0) {
      numLosingTrades += 1;
    } else if (trades[i].percent_result > 0) {
      numWinningTrades += 1;
    }
  }

  const worstTrade = trades[0];
  const bestTrade = trades[trades.length - 1];
  const totalTrades = numLosingTrades + numWinningTrades;

  return {
    chartData: ret,
    mean: roundNumberDropRemaining(
      safeDivide(cumulativeResults, ret.length, 0),
      2
    ),
    worstTradePerc: worstTrade !== undefined ? worstTrade.percent_result : null,
    bestTradePerc: bestTrade !== undefined ? bestTrade.percent_result : null,
    shareOfAllTrades: roundNumberDropRemaining(
      safeDivide(totalTrades, trades.length, 0) * 100,
      2
    ),
    winningTradesRatio: roundNumberDropRemaining(
      safeDivide(numWinningTrades, totalTrades, 0) * 100,
      2
    ),
    losingTradesRatio: roundNumberDropRemaining(
      safeDivide(numLosingTrades, totalTrades, 0) * 100,
      2
    ),
  };
};

export const getDateRange = (portfolioTicks: ChartTick[]): string => {
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

const LongShortSummaryCard = ({ backtest }: { backtest: BacktestObject }) => {
  return (
    <ChakraCard heading={<Heading size={"md"}>Trading rules</Heading>}>
      <div style={{ marginTop: "16px" }}>
        <Heading size={"md"}>Assumptions</Heading>

        <div style={{ marginTop: "16px" }}>
          <div style={{ marginTop: "8px" }}>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              Use time based close:{" "}
              {backtest.use_time_based_close
                ? `True, ${backtest.klines_until_close} candles`
                : "False"}
            </pre>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              Use stop loss based close:{" "}
              {backtest.use_stop_loss_based_close
                ? `True, ${backtest.stop_loss_threshold_perc}%`
                : "False"}
            </pre>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              Use profit (%) based close:{" "}
              {backtest.use_profit_based_close
                ? `True, ${backtest.take_profit_threshold_perc}%`
                : "False"}
            </pre>
          </div>
        </div>
      </div>
      <div style={{ marginTop: "16px" }}>
        <Heading size={"md"}>Buy and sell rules</Heading>
        <div style={{ marginTop: "16px" }}>
          <div style={{ marginTop: "8px" }}>
            <pre style={{ color: COLOR_CONTENT_PRIMARY }}>
              {backtest.long_short_buy_cond}
            </pre>
          </div>
          <div style={{ marginTop: "8px" }}>
            <pre style={{ color: COLOR_CONTENT_PRIMARY }}>
              {backtest.long_short_sell_cond}
            </pre>
          </div>
          <div style={{ marginTop: "8px" }}>
            <pre style={{ color: COLOR_CONTENT_PRIMARY }}>
              {backtest.long_short_exit_cond}
            </pre>
          </div>
        </div>
      </div>
    </ChakraCard>
  );
};

export const LongShortBacktestsDetailsView = () => {
  const { massPairTradeBacktestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(massPairTradeBacktestId));

  const [hideBenchmark, setHideBenchmark] = useState(false);
  const [filterTradesRange, setFilterTradesRange] = useState([-500, 500]);
  const deployFormDisclosure = useDisclosure();

  const portfolioTicks = useMemo(
    () => getPortfolioChartData(backtestQuery.data as FetchBacktestByIdRes),
    [backtestQuery.data]
  );

  const tradeDetailsData = useMemo(
    () =>
      getTradesData(
        backtestQuery.data as FetchBacktestByIdRes,
        filterTradesRange
      ),
    [backtestQuery.data, JSON.stringify(filterTradesRange)]
  );

  useEffect(() => {
    if (backtestQuery.data) {
      const trades = backtestQuery.data.pair_trades.sort(
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

  if (backtestQuery.isLoading || !backtestQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const backtest = backtestQuery.data.data;

  return (
    <>
      <ChakraDrawer
        {...deployFormDisclosure}
        title={"Deploy L/S strategy"}
        drawerContentStyles={{ maxWidth: "80%" }}
      >
        <DeployLongShortStrategyForm
          onSuccessCallback={() => {
            deployFormDisclosure.onClose();
          }}
        />
      </ChakraDrawer>
      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div>
            <Heading size={"lg"}>Pair-trade backtest {backtest.name}</Heading>
          </div>
          <div>
            <Button
              leftIcon={<GrDeploy />}
              onClick={deployFormDisclosure.onOpen}
            >
              Deploy
            </Button>
          </div>
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

        <div style={{ marginTop: "16px" }}>
          <div></div>
          <div style={{ marginTop: "16px" }}>
            <ChakraCard heading={<Heading size={"md"}>Trade results</Heading>}>
              <div
                style={{ display: "flex", alignItems: "center", gap: "16px" }}
              >
                <div>
                  <Stat color={COLOR_CONTENT_PRIMARY}>
                    <StatLabel>Result mean</StatLabel>
                    <StatNumber>{tradeDetailsData["mean"]}%</StatNumber>
                  </Stat>
                </div>
                <div>
                  <Stat color={COLOR_CONTENT_PRIMARY}>
                    <StatLabel>Winning trades</StatLabel>
                    <StatNumber>
                      {tradeDetailsData["winningTradesRatio"]}%
                    </StatNumber>
                  </Stat>
                </div>
                <div>
                  <Stat color={COLOR_CONTENT_PRIMARY}>
                    <StatLabel>Losing trades</StatLabel>
                    <StatNumber>
                      {tradeDetailsData["losingTradesRatio"]}%
                    </StatNumber>
                  </Stat>
                </div>
                <div>
                  <Stat color={COLOR_CONTENT_PRIMARY}>
                    <StatLabel>Share of all trades</StatLabel>
                    <StatNumber>
                      {tradeDetailsData["shareOfAllTrades"]}%
                    </StatNumber>
                  </Stat>
                </div>
                <div>
                  <Stat color={COLOR_CONTENT_PRIMARY}>
                    <StatLabel>Num trades</StatLabel>
                    <StatNumber>
                      {tradeDetailsData["chartData"].length}
                    </StatNumber>
                  </Stat>
                </div>
              </div>
            </ChakraCard>
          </div>
          <div style={{ width: "500px", marginTop: "16px" }}>
            <GenericRangeSlider
              minValue={tradeDetailsData["worstTradePerc"]}
              maxValue={tradeDetailsData["bestTradePerc"]}
              values={filterTradesRange}
              onChange={(newValues: number[]) => {
                setFilterTradesRange(newValues);
              }}
              formatLabelCallback={(values) =>
                `Trades from ${roundNumberDropRemaining(
                  values[0],
                  2
                )}% to ${roundNumberDropRemaining(values[1], 2)}%`
              }
            />
          </div>
          <div>
            <GenericBarChart
              data={tradeDetailsData["chartData"]}
              yAxisKey="perc_result"
              xAxisKey=""
              containerStyles={{ marginTop: "16px" }}
            />
          </div>
        </div>
        <LongShortSummaryCard backtest={backtest} />
      </div>
    </>
  );
};
