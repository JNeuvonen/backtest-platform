import React, { useMemo, useState } from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useManyBacktests,
  useMassbacktest,
} from "../../../clients/queries/queries";
import { Checkbox, Spinner, Switch } from "@chakra-ui/react";
import { FetchBulkBacktests } from "../../../clients/queries/response-types";
import { LINE_CHART_COLORS, MAX_NUMBER_OF_LINES } from "../../../utils/algo";
import { ShareYAxisMultilineChart } from "../../../components/charts/ShareYAxisMultiline";
import { Line } from "recharts";
import { useMessageListener } from "../../../hooks/useMessageListener";
import { DOM_EVENT_CHANNELS } from "../../../utils/constants";
import { ChakraSelect } from "../../../components/chakra/Select";
import {
  COMBINED_STRATEGY_DATA_KEY,
  getMassSimEquityCurvesData,
} from "../../../utils/backtest";
import { WithLabel } from "../../../components/form/WithLabel";

interface PathParams {
  massBacktestId: number;
}

const FILTER_NOT_SELECTED_VALUE = "not-selected";
const FILTER_NOT_SELECTED_LABEL = "Unselected";

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

  const [showOnlyCombinedEqFilter, setShowOnlyCombinedEqFilter] =
    useState(false);

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
      { selectedYearFilter, sinceYearFilter, FILTER_NOT_SELECTED_VALUE }
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
          <div>
            <WithLabel label={"Show only combined eq"}>
              <Switch
                isChecked={showOnlyCombinedEqFilter}
                onChange={() =>
                  setShowOnlyCombinedEqFilter(!showOnlyCombinedEqFilter)
                }
              />
            </WithLabel>
          </div>
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
          {showOnlyCombinedEqFilter
            ? null
            : datasetSymbols.map((item, idx) => {
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
          <Line
            type="monotone"
            dataKey={COMBINED_STRATEGY_DATA_KEY}
            stroke={"red"}
            dot={false}
          />
        </ShareYAxisMultilineChart>
      </div>
    </div>
  );
};
