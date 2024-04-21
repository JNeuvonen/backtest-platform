import React, { useEffect, useMemo, useState } from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useManyBacktests,
  useMassbacktest,
} from "../../../clients/queries/queries";
import {
  Button,
  Checkbox,
  Heading,
  Spinner,
  Switch,
  useDisclosure,
} from "@chakra-ui/react";
import { FetchBulkBacktests } from "../../../clients/queries/response-types";
import { LINE_CHART_COLORS, MAX_NUMBER_OF_LINES } from "../../../utils/algo";
import { ShareYAxisMultilineChart } from "../../../components/charts/ShareYAxisMultiline";
import { Line } from "recharts";
import { useMessageListener } from "../../../hooks/useMessageListener";
import { DOM_EVENT_CHANNELS } from "../../../utils/constants";
import { ChakraSelect } from "../../../components/chakra/Select";
import {
  COMBINED_STRATEGY_DATA_KEY,
  getBulkBacktestDetails,
  getDatasetsInBulkBacktest,
} from "../../../utils/backtest";
import { WithLabel } from "../../../components/form/WithLabel";
import { ChakraPopover } from "../../../components/chakra/popover";
import { useForceUpdate } from "../../../hooks/useForceUpdate";
import { BUTTON_VARIANTS } from "../../../theme";

interface PathParams {
  massBacktestId: number;
}

const FILTER_NOT_SELECTED_VALUE = "not-selected";
const FILTER_NOT_SELECTED_LABEL = "Unselected";

export interface DisplayPairsItem {
  datasetSymbol: string;
  display: boolean;
}

const SelectDatasetsPopoverBody = ({
  datasets,
}: {
  datasets: DisplayPairsItem[];
}) => {
  const forceUpdate = useForceUpdate();
  return (
    <div style={{ maxHeight: "400px", overflowY: "auto" }}>
      <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
        <Button
          variant={BUTTON_VARIANTS.nofill}
          fontSize={"14px"}
          onClick={() => {
            datasets.map((item) => {
              item.display = true;
            });

            forceUpdate();
          }}
        >
          Select all
        </Button>
        <Button
          variant={BUTTON_VARIANTS.nofill}
          fontSize={"14px"}
          onClick={() => {
            datasets.map((item) => {
              item.display = false;
            });
            forceUpdate();
          }}
        >
          Unselect all
        </Button>
      </div>
      <div
        style={{
          display: "flex",
          gap: "16px",
          flexDirection: "column",
          marginTop: "16px",
        }}
      >
        {datasets.map((item) => {
          return (
            <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
              <Checkbox
                isChecked={item.display}
                onChange={() => {
                  datasets.map((datasetItem) => {
                    if (datasetItem.datasetSymbol === item.datasetSymbol) {
                      datasetItem.display = !datasetItem.display;
                    }
                  });
                  forceUpdate();
                }}
              />
              <div>{item.datasetSymbol}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
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
  const selectPairsPopover = useDisclosure();

  const [showOnlyCombinedEqFilter, setShowOnlyCombinedEqFilter] =
    useState(false);

  const [currentlyDisplayedPairs, setCurrentlyDisplayedPairs] = useState<
    DisplayPairsItem[]
  >([]);

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_component,
    messageCallback: () => {
      massBacktestQuery.refetch();
      useManyBacktestsQuery.refetch();
    },
  });

  useEffect(() => {
    if (useManyBacktestsQuery.data) {
      const datasets = getDatasetsInBulkBacktest(useManyBacktestsQuery.data);
      setCurrentlyDisplayedPairs(
        datasets.map((item) => {
          return {
            display: true,
            datasetSymbol: item,
          };
        })
      );
    }
  }, [useManyBacktestsQuery.data]);

  const bulkBacktestDetails = useMemo(() => {
    return getBulkBacktestDetails(
      useManyBacktestsQuery.data as FetchBulkBacktests,
      {
        selectedYearFilter,
        sinceYearFilter,
        FILTER_NOT_SELECTED_VALUE,
      }
    );
  }, [useManyBacktestsQuery.data, selectedYearFilter, sinceYearFilter]);

  if (
    massBacktestQuery.isLoading ||
    !massBacktestQuery.data ||
    useManyBacktestsQuery.isLoading ||
    !useManyBacktestsQuery.data ||
    bulkBacktestDetails === null
  ) {
    return <Spinner />;
  }

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
            <ChakraPopover
              {...selectPairsPopover}
              setOpen={selectPairsPopover.onOpen}
              body={
                <SelectDatasetsPopoverBody datasets={currentlyDisplayedPairs} />
              }
              headerText="Selected pairs"
            >
              <Button variant={BUTTON_VARIANTS.nofill}>Select pairs</Button>
            </ChakraPopover>
          </div>
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
              ...bulkBacktestDetails["years"].map((item) => ({
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
              ...bulkBacktestDetails["years"].map((item) => ({
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
          data={
            bulkBacktestDetails === null
              ? []
              : bulkBacktestDetails["equityCurves"]
          }
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
            : bulkBacktestDetails?.datasets.map((item, idx) => {
                const currentlyDisplayed = currentlyDisplayedPairs.filter(
                  (dataset) => dataset.datasetSymbol === item
                );

                if (
                  currentlyDisplayed.length > 0 &&
                  !currentlyDisplayed[0].display
                ) {
                  return null;
                }

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
