import React from "react";
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
  getLineChartColors,
} from "../../../utils/algo";
import { ShareYAxisMultilineChart } from "../../../components/charts/ShareYAxisMultiline";
import { Line } from "recharts";
import { COLOR_BRAND_PRIMARY } from "../../../utils/colors";
import { useMessageListener } from "../../../hooks/useMessageListener";
import { DOM_EVENT_CHANNELS } from "../../../utils/constants";

interface PathParams {
  massBacktestId: number;
}

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

const getMassSimEquityCurvesData = (bulkFetchBacktest: FetchBulkBacktests) => {
  const ret: object[] = [];

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

  return ret;
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

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_component,
    messageCallback: () => {
      massBacktestQuery.refetch();
      useManyBacktestsQuery.refetch();
    },
  });

  if (
    massBacktestQuery.isLoading ||
    !massBacktestQuery.data ||
    useManyBacktestsQuery.isLoading ||
    !useManyBacktestsQuery.data
  ) {
    return <Spinner />;
  }

  const equityCurves = getMassSimEquityCurvesData(useManyBacktestsQuery.data);
  const datasetSymbols = getLineKeys(useManyBacktestsQuery.data);

  return (
    <div>
      <ShareYAxisMultilineChart
        height={500}
        data={equityCurves}
        xAxisKey={"kline_open_time"}
        xAxisTickFormatter={(tick: number) =>
          new Date(tick).toLocaleDateString()
        }
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
            />
          );
        })}
      </ShareYAxisMultilineChart>
    </div>
  );
};
