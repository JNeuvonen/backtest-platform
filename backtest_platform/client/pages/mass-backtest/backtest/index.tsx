import React, { useMemo, useState } from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useManyBacktests,
  useMassbacktest,
} from "../../../clients/queries/queries";
import { Spinner } from "@chakra-ui/react";
import { FetchBulkBacktests } from "../../../clients/queries/response-types";
import {
  LINE_CHART_COLORS,
  MAX_NUMBER_OF_LINES,
  binarySearch,
} from "../../../utils/algo";
import { ShareYAxisMultilineChart } from "../../../components/charts/ShareYAxisMultiline";
import { Line } from "recharts";
import { useMessageListener } from "../../../hooks/useMessageListener";
import { DOM_EVENT_CHANNELS } from "../../../utils/constants";
import { isFinite } from "lodash";
import { ChakraSelect } from "../../../components/chakra/Select";
import { getKeysCount } from "../../../utils/object";

interface PathParams {
  massBacktestId: number;
}

const FILTER_NOT_SELECTED_VALUE = "not-selected";
const FILTER_NOT_SELECTED_LABEL = "Unselected";

const getMassSimFindTicks = (
  bulkFetchBacktest: FetchBulkBacktests,
  klineOpenTime: number
) => {
  const ret = {};

  bulkFetchBacktest.equity_curves.forEach((item) => {
    for (const [key, equityCurveTicks] of Object.entries(item)) {
      const item = binarySearch(
        equityCurveTicks,
        klineOpenTime,
        (tick) => tick.kline_open_time
      );

      ret[bulkFetchBacktest.id_to_dataset_name_map[key]] =
        item?.portfolio_worth || null;
    }
  });
  return ret;
};

const convertToReturn = (currItem: object, prevItem: object, key: string) => {
  const currVal = currItem[key];
  const prevVal = prevItem[key];

  if (prevVal === 0) {
    return 1;
  }

  const value = currVal / prevVal;

  if (isNaN(value) || !isFinite(value)) {
    return 1;
  }

  return value;
};

const getEquityCurveStatistics = (
  bulkFetchBacktest: FetchBulkBacktests,
  balances: object[],
  returns: object[],
  backtestKeyWithMostKlines: string
) => {
  const endBalances = {};

  bulkFetchBacktest.equity_curves.forEach((item) => {
    for (const [key, equityCurveTicks] of Object.entries(item)) {
      if (key === "kline_open_time") {
        continue;
      }
      for (let i = equityCurveTicks.length - 1; i >= 0; --i) {
        const tick = equityCurveTicks[i];

        if (tick.portfolio_worth) {
          endBalances[bulkFetchBacktest.id_to_dataset_name_map[key]] =
            tick.portfolio_worth;
          break;
        }
      }
    }
  });

  let multiStratCurrBalance = 10000;
  const multiStrategyEquityCurve: object[] = [];

  const totalStrats = getKeysCount(returns[0]) - 1;

  for (let i = 0; i < returns.length; ++i) {
    const returnsTick = returns[i];

    const tick = {};
    const roundReturns = {};

    for (const [key, value] of Object.entries(returnsTick)) {
      if (key === "kline_open_time") {
        tick["kline_open_time"] = value;
        continue;
      }
      const coeff = value === 0 ? 1 : value;

      roundReturns[key] = (multiStratCurrBalance / totalStrats) * coeff;
    }

    let tickBalance = 0;

    for (const [_, value] of Object.entries(roundReturns)) {
      tickBalance += value as number;
    }

    multiStratCurrBalance = tickBalance;
    tick["equity"] = multiStratCurrBalance;
    multiStrategyEquityCurve.push(tick);
  }

  return {
    endBalances: endBalances,
    multiStrategyEquityCurve: multiStrategyEquityCurve,
  };
};

const getMassSimEquityCurvesData = (
  bulkFetchBacktest: FetchBulkBacktests,
  {
    sinceYearFilter,
    selectedYearFilter,
  }: { sinceYearFilter: string; selectedYearFilter: string }
) => {
  if (!bulkFetchBacktest || !bulkFetchBacktest.data) return null;

  const ret: object[] = [];
  const yearsUsedInBacktest = new Set();

  let backtestKeyWithMostKlines = "";
  let mostKlinesCount = 0;

  bulkFetchBacktest.equity_curves.forEach((item) => {
    for (const [key, value] of Object.entries(item)) {
      if (value.length > mostKlinesCount) {
        mostKlinesCount = value.length;
        backtestKeyWithMostKlines = key;
      }
    }
  });

  bulkFetchBacktest.equity_curves.forEach((item) => {
    for (const [key, equityCurveTicks] of Object.entries(item)) {
      if (key === backtestKeyWithMostKlines) {
        equityCurveTicks.forEach((balance) => {
          const timestampToDate = new Date(balance["kline_open_time"]);
          yearsUsedInBacktest.add(timestampToDate.getFullYear());

          if (
            selectedYearFilter !== FILTER_NOT_SELECTED_VALUE &&
            String(timestampToDate.getFullYear()) !== selectedYearFilter
          ) {
            return;
          }

          if (
            sinceYearFilter !== FILTER_NOT_SELECTED_VALUE &&
            timestampToDate.getFullYear() < Number(sinceYearFilter)
          ) {
            return;
          }

          const portfolioValues = getMassSimFindTicks(
            bulkFetchBacktest,
            balance.kline_open_time
          );

          ret.push({
            kline_open_time: balance.kline_open_time,
            ...portfolioValues,
          });
        });
      }
    }
  });

  const returns = [] as object[];

  for (let i = 1; i < ret.length; ++i) {
    const prevItem = ret[i - 1];
    const currItem = ret[i];

    const tick = {
      kline_open_time: currItem["kline_open_time"],
    };

    for (const [key, _] of Object.entries(currItem)) {
      if (key === "kline_open_time") {
        continue;
      }
      const value = convertToReturn(currItem, prevItem, key);
      tick[key] = value;
    }
    returns.push(tick);
  }

  const final = [] as object[];
  const helper = {};

  for (let i = 0; i < returns.length; ++i) {
    const currItem = returns[i];
    if (i === 0) {
      for (const [key] of Object.entries(currItem)) {
        if (key === "kline_open_time") {
          continue;
        }
        helper[key] = 1;
      }
      continue;
    }

    const tick = {
      kline_open_time: currItem["kline_open_time"],
    };

    for (const [key, value] of Object.entries(currItem)) {
      if (key === "kline_open_time") {
        continue;
      }
      if (value === undefined || value === null || isNaN(value) || value == 0) {
        tick[key] = (helper[key] - 1) * 100;
        continue;
      }
      helper[key] = helper[key] * value;
      tick[key] = (helper[key] - 1) * 100;
    }

    final.push(tick);
  }

  const eqCurveStatistics = getEquityCurveStatistics(
    bulkFetchBacktest,
    ret,
    returns,
    backtestKeyWithMostKlines
  );

  return {
    equityCurves: final,
    years: Array.from(yearsUsedInBacktest),
    ...eqCurveStatistics,
  };
};

const getLineKeys = (bulkFetchBacktest: FetchBulkBacktests) => {
  const ret = [] as string[];

  for (const [_, value] of Object.entries(
    bulkFetchBacktest.id_to_dataset_name_map
  )) {
    ret.push(value);
  }
  return ret;
};

export const InvidualMassbacktestDetailsPage = () => {
  const { massBacktestId } = usePathParams<PathParams>();
  const massBacktestQuery = useMassbacktest(Number(massBacktestId));
  const useManyBacktestsQuery = useManyBacktests(
    massBacktestQuery.data?.backtest_ids || [],
    true
  );
  const [selectedYearFilter, setSelectedYearFilter] = useState(
    FILTER_NOT_SELECTED_VALUE
  );
  const [sinceYearFilter, setSinceYearFilter] = useState(
    FILTER_NOT_SELECTED_VALUE
  );

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_component,
    messageCallback: () => {
      massBacktestQuery.refetch();
      useManyBacktestsQuery.refetch();
    },
  });

  const equityCurves = useMemo(() => {
    return getMassSimEquityCurvesData(
      useManyBacktestsQuery.data as FetchBulkBacktests,
      { selectedYearFilter, sinceYearFilter }
    );
  }, [useManyBacktestsQuery.data, selectedYearFilter, sinceYearFilter]);

  if (
    massBacktestQuery.isLoading ||
    !massBacktestQuery.data ||
    useManyBacktestsQuery.isLoading ||
    !useManyBacktestsQuery.data
  ) {
    return <Spinner />;
  }

  const datasetSymbols = getLineKeys(useManyBacktestsQuery.data);

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div></div>

        <div style={{ gap: "16px", display: "flex", alignItems: "center" }}>
          <ChakraSelect
            label={"Since year"}
            containerStyle={{ width: "150px" }}
            options={[
              {
                label: FILTER_NOT_SELECTED_LABEL,
                value: FILTER_NOT_SELECTED_VALUE,
              },
              ...equityCurves["years"].map((item) => ({
                label: String(item),
                value: String(item),
              })),
            ]}
            onChange={(value: string) => {
              setSelectedYearFilter(FILTER_NOT_SELECTED_VALUE);
              setSinceYearFilter(value);
            }}
          />
          <ChakraSelect
            label={"Filter by year"}
            containerStyle={{ width: "150px" }}
            options={[
              {
                label: FILTER_NOT_SELECTED_LABEL,
                value: FILTER_NOT_SELECTED_VALUE,
              },
              ...equityCurves["years"].map((item) => ({
                label: String(item),
                value: String(item),
              })),
            ]}
            onChange={(value: string) => {
              setSinceYearFilter(FILTER_NOT_SELECTED_VALUE);
              setSelectedYearFilter(value);
            }}
          />
        </div>
      </div>
      <div style={{ marginTop: "16px" }}>
        <ShareYAxisMultilineChart
          height={500}
          data={equityCurves === null ? [] : equityCurves["equityCurves"]}
          xAxisKey={"kline_open_time"}
          xAxisTickFormatter={(tick: number) =>
            new Date(tick).toLocaleDateString("default", {
              year: "numeric",
              month: "short",
            })
          }
          yAxisTickFormatter={(value: number) => `${value}%`}
        >
          {datasetSymbols.map((item, idx) => {
            return (
              <Line
                type="monotone"
                dataKey={item}
                stroke={
                  idx > MAX_NUMBER_OF_LINES
                    ? LINE_CHART_COLORS[idx % MAX_NUMBER_OF_LINES]
                    : LINE_CHART_COLORS[idx]
                }
                dot={false}
                key={item}
              />
            );
          })}
        </ShareYAxisMultilineChart>
      </div>
      <div style={{ marginTop: "16px" }}>
        <ShareYAxisMultilineChart
          height={500}
          data={
            equityCurves === null
              ? []
              : equityCurves["multiStrategyEquityCurve"]
          }
          xAxisKey={"kline_open_time"}
          xAxisTickFormatter={(tick: number) =>
            new Date(tick).toLocaleDateString("default", {
              year: "numeric",
              month: "short",
            })
          }
        >
          <Line type="monotone" dataKey={"equity"} stroke={"red"} dot={false} />
        </ShareYAxisMultilineChart>
      </div>
    </div>
  );
};
