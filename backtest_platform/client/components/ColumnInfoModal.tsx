import React, { useEffect, useState } from "react";
import { useColumnQuery } from "../clients/queries/queries";
import { ChakraCard } from "./chakra/Card";
import {
  Box,
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
} from "@chakra-ui/react";
import { CARD_VARIANTS } from "../theme";
import { COLOR_BRAND_SECONDARY, COLOR_CONTENT_PRIMARY } from "../utils/colors";
import {
  getNormalDistributionItems,
  roundNumberDropRemaining,
} from "../utils/number";
import { ColumnChart } from "./charts/column";
import { createColumnChartData } from "../utils/dataset";
import { WithLabel } from "./form/WithLabel";
import { GenericBarChart } from "./charts/BarChart";
import { parseJson } from "../utils/str";

interface Props {
  datasetName: string;
  columnName: string;
}

export const ColumnInfoModal = (props: Props) => {
  const { datasetName, columnName } = props;
  const columnDetailedQuery = useColumnQuery(datasetName, columnName);
  const [linearRegrPltSrc, setLinearRegrPltSrc] = useState("");

  const columnData = columnDetailedQuery.data?.res.column;

  useEffect(() => {
    if (columnDetailedQuery.data?.res) {
      const data = columnDetailedQuery.data?.res;
      setLinearRegrPltSrc(`data:image/png;base64,${data.linear_regr_img_b64}`);
    }
  }, [columnDetailedQuery.data]);

  if (!columnData) return <Spinner />;

  const linearRegrSummary = columnDetailedQuery.data?.res.linear_regr_summary;
  const linearRegrParams = parseJson(
    columnDetailedQuery.data?.res.linear_regr_params || ""
  );

  return (
    <div>
      <ChakraCard
        heading={<Heading size="md">Stats</Heading>}
        variant={CARD_VARIANTS.on_modal}
      >
        <Box display={"flex"} alignItems={"center"} gap={"16px"}>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Corr to target</StatLabel>
              <StatNumber>
                {columnData.corr_to_price
                  ? String(
                      roundNumberDropRemaining(
                        columnData.corr_to_price * 100,
                        2
                      )
                    ) + "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </Box>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Mean</StatLabel>
              <StatNumber>
                {columnData.stats
                  ? String(
                      roundNumberDropRemaining(columnData.stats.mean, 4, true)
                    )
                  : "N/A"}
              </StatNumber>
            </Stat>
          </Box>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Median</StatLabel>
              <StatNumber>
                {columnData.stats
                  ? String(
                      roundNumberDropRemaining(columnData.stats.median, 4, true)
                    )
                  : "N/A"}
              </StatNumber>
            </Stat>
          </Box>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Std dev</StatLabel>
              <StatNumber>
                {columnData.stats
                  ? String(
                      roundNumberDropRemaining(
                        columnData.stats.std_dev,
                        4,
                        true
                      )
                    )
                  : "N/A"}
              </StatNumber>
            </Stat>
          </Box>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Max</StatLabel>
              <StatNumber>
                {columnData.stats
                  ? String(
                      roundNumberDropRemaining(columnData.stats.max, 4, true)
                    )
                  : "N/A"}
              </StatNumber>
            </Stat>
          </Box>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Min</StatLabel>
              <StatNumber>
                {columnData.stats
                  ? String(
                      roundNumberDropRemaining(columnData.stats.min, 4, true)
                    )
                  : "N/A"}
              </StatNumber>
            </Stat>
          </Box>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Linear regr const</StatLabel>
              <StatNumber>
                {columnData.corr_to_price
                  ? String(roundNumberDropRemaining(linearRegrParams.const, 6))
                  : "N/A"}
              </StatNumber>
            </Stat>
          </Box>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Linear regr coeff</StatLabel>
              <StatNumber>
                {columnData.corr_to_price
                  ? String(
                      roundNumberDropRemaining(linearRegrParams[columnName], 6)
                    )
                  : "N/A"}
              </StatNumber>
            </Stat>
          </Box>
        </Box>
      </ChakraCard>

      <Box marginTop={"16px"}>
        {linearRegrPltSrc && (
          <img src={linearRegrPltSrc} alt="Linear Regression Plot" />
        )}
      </Box>

      <Box marginTop={"32px"}>
        <ColumnChart
          data={createColumnChartData(
            columnData.rows,
            columnName,
            columnData.kline_open_time,
            columnData.price_data
          )}
          xAxisDataKey={"kline_open_time"}
          lines={[
            { dataKey: columnName, stroke: "red", yAxisId: "left" },
            {
              dataKey: "price",
              stroke: COLOR_BRAND_SECONDARY,
              yAxisId: "right",
            },
          ]}
        />
      </Box>

      <WithLabel
        label="Normal distribution"
        containerStyles={{ marginTop: "16px" }}
      >
        <GenericBarChart
          data={getNormalDistributionItems(columnData.rows)}
          yAxisKey="count"
          xAxisKey="label"
          containerStyles={{ marginTop: "16px" }}
        />
      </WithLabel>

      <ChakraCard
        heading={<Heading size="md">Linear regression analysis</Heading>}
        variant={CARD_VARIANTS.on_modal}
      >
        <Box>
          <pre style={{ color: COLOR_CONTENT_PRIMARY }}>
            {linearRegrSummary}
          </pre>
        </Box>
      </ChakraCard>
    </div>
  );
};
