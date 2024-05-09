import React, { useMemo, useState } from "react";
import { usePathParams } from "../../hooks/usePathParams";
import {
  useBacktestById,
  useMLModelsColumns,
  useModelQuery,
} from "../../clients/queries/queries";
import {
  Alert,
  AlertIcon,
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
  Switch,
  useDisclosure,
} from "@chakra-ui/react";
import ExternalLink from "../../components/ExternalLink";
import { getDatasetInfoPagePath } from "../../utils/navigate";
import { BacktestSummaryCard } from "../simulate/dataset/backtest/SummaryCard";
import {
  BacktestBalance,
  FetchBacktestByIdRes,
  Trade,
} from "../../clients/queries/response-types";
import { ShareYAxisMultilineChart } from "../../components/charts/ShareYAxisMultiline";
import { Line, Scatter } from "recharts";
import {
  COLOR_BRAND_SECONDARY,
  COLOR_CONTENT_PRIMARY,
} from "../../utils/colors";
import { WithLabel } from "../../components/form/WithLabel";
import { GenericBarChart } from "../../components/charts/BarChart";
import { TradesBarChartData } from "../simulate/dataset/backtest";
import { TradingCriteriaCard } from "../simulate/dataset/backtest/TradingCriteriaCard";
import { ChakraModal } from "../../components/chakra/modal";
import { ColumnInfoModal } from "../../components/ColumnInfoModal";
import { ChakraInput } from "../../components/chakra/input";
import { getDateRange } from "../simulate/bulk/backtest-details-view";
import { ChakraCard } from "../../components/chakra/Card";
import { roundNumberDropRemaining, safeDivide } from "../../utils/number";
import { formatSecondsIntoTime } from "../../utils/date";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

const getTradesData = (trades: Trade[]) => {
  const ret = [] as TradesBarChartData[];

  const percResultSorted = trades.sort(
    (a, b) => a.percent_result - b.percent_result
  );
  for (let i = 0; i < percResultSorted.length; i++) {
    ret.push({
      perc_result: percResultSorted[i].percent_result,
      kline_open_time: percResultSorted[i].open_time,
    });
  }

  return ret;
};

const findTickBasedOnOpenTime = (
  ticks: BacktestBalance[],
  kline_open_time: number
) => {
  const tick = ticks.filter((item) => item.kline_open_time === kline_open_time);
  return tick.length > 0 ? tick[0] : null;
};

const getPortfolioGrowthData = (backtestData: FetchBacktestByIdRes) => {
  const ret = [] as object[];

  const portfolioData = backtestData.balance_history;
  const increment = Math.max(Math.floor(portfolioData.length / 1000), 1);

  const trades = backtestData.trades;

  const setOfUsedKlines = new Set();

  for (let i = 0; i < trades.length; i++) {
    const isShortTrade = trades[i].is_short_trade;
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
      kline_open_time: tickEnter["kline_open_time"] * 1000,
      buy_and_hold: tickClose["benchmark_price"],
      enter_short: isShortTrade ? tickEnter["benchmark_price"] : null,
      enter_long: isShortTrade ? null : tickEnter["benchmark_price"],
      exit_long: null,
      exit_short: null,
    });
    ret.push({
      strategy: tickClose["portfolio_worth"],
      kline_open_time: tickClose["kline_open_time"] * 1000,
      buy_and_hold: tickClose["benchmark_price"],
      enter_short: null,
      enter_long: null,
      exit_long: isShortTrade ? null : tickClose["benchmark_price"],
      exit_short: isShortTrade ? tickClose["benchmark_price"] : null,
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
      kline_open_time: item["kline_open_time"] * 1000,
      buy_and_hold: item["benchmark_price"],
      enter_short: null,
      enter_long: null,
      exit_long: null,
      exit_short: null,
    });
  }

  ret.sort((a, b) => a.kline_open_time - b.kline_open_time);

  return ret;
};

interface MLModelsTrainColumnsProps {
  datasetName: string;
  modelId: number;
}

const MLModelsTrainColumns = ({
  datasetName,
  modelId,
}: MLModelsTrainColumnsProps) => {
  const modelColumnsQuery = useMLModelsColumns(modelId);
  const columnInfoModal = useDisclosure();
  const [selectedColumn, setSelectedColumn] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  if (!modelColumnsQuery.data) return <Spinner />;

  return (
    <div style={{ marginTop: "16px" }}>
      <ChakraModal
        {...columnInfoModal}
        title={`Column info ${selectedColumn}`}
        modalContentStyle={{ maxWidth: "80%", marginTop: "5%" }}
      >
        <ColumnInfoModal
          datasetName={datasetName}
          columnName={selectedColumn}
        />
      </ChakraModal>

      <ChakraInput
        label={"Filter columns"}
        value={searchTerm}
        onChange={(value) => setSearchTerm(value)}
      />

      <div style={{ maxHeight: "500px", overflowY: "auto", marginTop: "8px" }}>
        {modelColumnsQuery.data
          .filter((item) => {
            if (searchTerm) {
              return item.toLowerCase().includes(searchTerm.toLowerCase());
            }
            return true;
          })
          .map((item) => {
            return (
              <div
                key={item}
                className="link-default"
                onClick={() => {
                  setSelectedColumn(item);
                  columnInfoModal.onOpen();
                }}
                style={{
                  width: "max-content",
                }}
              >
                {item}
              </div>
            );
          })}
      </div>
    </div>
  );
};

const TradesSummaryCard = ({
  trades,
  title,
}: {
  trades: Trade[];
  title: string;
}) => {
  const calculateTradeDetails = () => {
    let grossProfits = 0;
    let grossLosses = 0;
    let winningTrades = 0;
    let losingTrades = 0;
    let cumulativeHoldTime = 0;
    let cumulativePercResult = 0.0;
    let bestTradeResPerc = -1000;
    let worstTradeResPerc = 1000;

    trades.forEach((item) => {
      cumulativeHoldTime += item.close_time - item.open_time;
      cumulativePercResult += item.percent_result;

      if (item.net_result > 0) {
        grossProfits += item.net_result;
        winningTrades += 1;
      }

      if (item.net_result < 0) {
        grossLosses += Math.abs(item.net_result);
        losingTrades += 1;
      }

      if (item.percent_result < worstTradeResPerc) {
        worstTradeResPerc = item.percent_result;
      }

      if (item.percent_result > bestTradeResPerc) {
        bestTradeResPerc = item.percent_result;
      }
    });

    const numTrades = trades.length;
    const meanHoldTimeSec = safeDivide(cumulativeHoldTime, numTrades, 0);

    return {
      profitFactor: safeDivide(grossProfits, grossLosses, 0),
      meanTradePercRes: safeDivide(cumulativePercResult, numTrades, 0),
      meanHoldTime: formatSecondsIntoTime(meanHoldTimeSec),
      worstTradeResult: worstTradeResPerc,
      bestTradeResult: bestTradeResPerc,
      tradeCount: numTrades,
    };
  };

  const tradeDetailsDict = useMemo(() => {
    return calculateTradeDetails();
  }, [trades]);

  return (
    <ChakraCard heading={<Heading size="md">{title}</Heading>}>
      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>Profit factor</StatLabel>
            <StatNumber>
              {roundNumberDropRemaining(tradeDetailsDict.profitFactor, 2)}
            </StatNumber>
          </Stat>
        </div>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>Mean result (%)</StatLabel>
            <StatNumber>
              {roundNumberDropRemaining(tradeDetailsDict.meanTradePercRes, 2)}
            </StatNumber>
          </Stat>
        </div>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>Mean hold time</StatLabel>
            <StatNumber>{tradeDetailsDict.meanHoldTime}</StatNumber>
          </Stat>
        </div>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>Num trades</StatLabel>
            <StatNumber>{tradeDetailsDict.tradeCount}</StatNumber>
          </Stat>
        </div>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>Best trade</StatLabel>
            <StatNumber>
              {roundNumberDropRemaining(tradeDetailsDict.bestTradeResult, 2)}%
            </StatNumber>
          </Stat>
        </div>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>Worst trade</StatLabel>
            <StatNumber>
              {roundNumberDropRemaining(tradeDetailsDict.worstTradeResult, 2)}%
            </StatNumber>
          </Stat>
        </div>
      </div>
    </ChakraCard>
  );
};

export const MLBasedBacktestPage = () => {
  const { backtestId, datasetName } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));
  const modelQuery = useModelQuery(
    backtestQuery.data?.data.model_id || undefined
  );

  const [showLongEnters, setShowLongEnters] = useState(false);
  const [showLongExits, setShowLongExits] = useState(false);
  const [showShortEnters, setShowShortEnters] = useState(false);
  const [showShortExits, setShowShortExits] = useState(false);
  const [hideBenchmark, setHideBenchmark] = useState(false);
  const [hideStrategy, setHideStrategy] = useState(false);

  if (!backtestQuery.data || !backtestQuery.data.data || !modelQuery.data)
    return <Spinner />;

  const backtest = backtestQuery.data.data;

  const portfolioTicks = getPortfolioGrowthData(backtestQuery.data);
  const shortTrades = backtestQuery.data.trades.filter(
    (item) => item.is_short_trade
  );

  const longTrades = backtestQuery.data.trades.filter(
    (item) => !item.is_short_trade
  );

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
          {datasetName} {!backtestQuery.data.data.name}
        </Heading>
        <div style={{ display: "flex", gap: "16px" }}>
          <ExternalLink
            to={getDatasetInfoPagePath(datasetName)}
            linkText={"Model"}
          />
          <ExternalLink
            to={getDatasetInfoPagePath(datasetName)}
            linkText={"Dataset"}
          />
        </div>
      </div>

      <BacktestSummaryCard
        backtest={backtest}
        dateRange={getDateRange(portfolioTicks)}
      />

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "16px",
          marginTop: "16px",
        }}
      >
        <Heading size={"md"}>Backtest balance growth</Heading>

        <div style={{ display: "flex", gap: "6px" }}>
          <WithLabel
            label={"Hide strategy"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={hideStrategy}
              onChange={() => setHideStrategy(!hideStrategy)}
            />
          </WithLabel>
          <WithLabel
            label={"Hide benchmark"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={hideBenchmark}
              onChange={() => setHideBenchmark(!hideBenchmark)}
            />
          </WithLabel>
          <WithLabel
            label={"Long enters"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={showLongEnters}
              onChange={() => setShowLongEnters(!showLongEnters)}
            />
          </WithLabel>
          <WithLabel
            label={"Long exits"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={showLongExits}
              onChange={() => setShowLongExits(!showLongExits)}
            />
          </WithLabel>
          <WithLabel
            label={"Short enters"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={showShortEnters}
              onChange={() => setShowShortEnters(!showShortEnters)}
            />
          </WithLabel>
          <WithLabel
            label={"Short exits"}
            containerStyles={{ width: "max-content" }}
          >
            <Switch
              isChecked={showShortExits}
              onChange={() => setShowShortExits(!showShortExits)}
            />
          </WithLabel>
        </div>
      </div>

      {hideBenchmark &&
        (showShortExits ||
          showLongExits ||
          showShortEnters ||
          showLongEnters) && (
          <div style={{ marginBottom: "16px" }}>
            <Alert status="info" color="black">
              <AlertIcon />
              Trade enters and exits will be only displayed if benchmark is
              visible.
            </Alert>
          </div>
        )}

      <ShareYAxisMultilineChart
        height={500}
        data={getPortfolioGrowthData(backtestQuery.data)}
        xAxisKey={"kline_open_time"}
        xAxisTickFormatter={(tick: number) =>
          new Date(tick).toLocaleDateString("default", {
            year: "numeric",
            month: "short",
          })
        }
      >
        {!hideBenchmark && (
          <Line
            type="monotone"
            dataKey={"buy_and_hold"}
            stroke={"red"}
            dot={false}
          />
        )}

        {!hideStrategy && (
          <Line
            type="monotone"
            dataKey={"strategy"}
            stroke={COLOR_BRAND_SECONDARY}
            dot={false}
          />
        )}

        {!hideBenchmark && (
          <>
            {showLongEnters && <Scatter dataKey="enter_long" fill="green" />}
            {showLongExits && <Scatter dataKey="exit_long" fill="red" />}
            {showShortEnters && <Scatter dataKey="enter_short" fill="green" />}
            {showShortExits && <Scatter dataKey="exit_short" fill="red" />}
          </>
        )}
      </ShareYAxisMultilineChart>

      <div style={{ display: "flex", gap: "8px" }}>
        <div style={{ width: "50%" }}>
          <Heading size="md">Long trades</Heading>
          <GenericBarChart
            data={getTradesData(longTrades)}
            yAxisKey="perc_result"
            xAxisKey="kline_open_time"
            containerStyles={{ marginTop: "16px" }}
          />
        </div>
        <div style={{ width: "50%" }}>
          <Heading size="md">Short trades</Heading>
          <GenericBarChart
            data={getTradesData(shortTrades)}
            yAxisKey="perc_result"
            xAxisKey="kline_open_time"
            containerStyles={{ marginTop: "16px" }}
          />
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <div style={{ width: "50%" }}>
          <TradesSummaryCard trades={longTrades} title={"Long trades"} />
        </div>
        <div style={{ width: "50%" }}>
          <TradesSummaryCard trades={shortTrades} title={"Short trades"} />
        </div>
      </div>

      <MLModelsTrainColumns
        datasetName={datasetName}
        modelId={backtestQuery.data.data.model_id}
      />

      <TradingCriteriaCard
        backtestQuery={backtestQuery}
        modelQuery={modelQuery}
      />
    </div>
  );
};
