import React, { useEffect, useState } from "react";
import { useColumnQuery } from "../clients/queries/queries";
import { ChakraCard } from "./chakra/Card";
import {
  Box,
  Heading,
  Spinner,
  Stat,
  StatArrow,
  StatHelpText,
  StatLabel,
  StatNumber,
} from "@chakra-ui/react";
import { CARD_VARIANTS } from "../theme";
import { COLOR_CONTENT_PRIMARY } from "../utils/colors";
import { roundNumberDropRemaining } from "../utils/number";

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

  console.log(linearRegrPltSrc);

  return (
    <div>
      <ChakraCard
        heading={<Heading size="md">Stats</Heading>}
        variant={CARD_VARIANTS.on_modal}
      >
        <Box display={"flex"} alignItems={"center"} gap={"16px"}>
          <Box>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Corr to price</StatLabel>
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
        </Box>
      </ChakraCard>

      <Box marginTop={"16px"}>
        {linearRegrPltSrc && (
          <img src={linearRegrPltSrc} alt="Linear Regression Plot" />
        )}
      </Box>
    </div>
  );
};
