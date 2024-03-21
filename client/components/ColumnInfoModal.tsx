import React from "react";
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

  const columnData = columnDetailedQuery.data?.res.column;

  if (!columnData) return <Spinner />;

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

        <Box marginTop={"16px"}>
          <Heading size="s">Correlation to future prices</Heading>
          <Box
            display={"flex"}
            alignItems={"center"}
            gap={"16px"}
            marginTop={"8px"}
          >
            {columnData.corrs_to_shifted_prices.map((item) => {
              if (
                columnData.corr_to_price === null ||
                columnData.corr_to_price === undefined
              )
                return null;

              const comparedToDefaultCorr = Math.abs(
                item.corr / columnData.corr_to_price - 1
              );
              const isLarger = item.corr > columnData.corr_to_price;

              return (
                <Box key={item.label}>
                  <Stat color={COLOR_CONTENT_PRIMARY}>
                    <StatLabel>{item.label}</StatLabel>
                    <StatNumber>
                      {columnData.stats
                        ? String(
                            roundNumberDropRemaining(item.corr * 100, 2, true)
                          ) + "%"
                        : "N/A"}
                    </StatNumber>
                    <StatHelpText>
                      <StatArrow type={isLarger ? "increase" : "decrease"} />
                      {columnData.stats
                        ? String(
                            roundNumberDropRemaining(
                              comparedToDefaultCorr * 100,
                              2,
                              true
                            ) + "%"
                          )
                        : "N/A"}
                    </StatHelpText>
                  </Stat>
                </Box>
              );
            })}
          </Box>
        </Box>
      </ChakraCard>
    </div>
  );
};
