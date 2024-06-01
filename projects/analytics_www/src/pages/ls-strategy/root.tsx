import {
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
  Text,
} from "@chakra-ui/react";
import {
  getDiffToPresentFormatted,
  getLongShortGroupTradeInfo,
  getNumberDisplayColor,
  roundNumberFloor,
  safeDivide,
} from "common_js";
import { ChakraCard } from "src/components/chakra";
import { usePathParams } from "src/hooks";
import {
  useBinanceSpotPriceInfo,
  useLatestBalanceSnapshot,
  useLongshortGroup,
} from "src/http/queries";
import { COLOR_CONTENT_PRIMARY } from "src/theme";

export const LsStrategyPage = () => {
  const { strategyName } = usePathParams<{ strategyName: string }>();
  const longshortGroupQuery = useLongshortGroup(strategyName);
  const binancePriceQuery = useBinanceSpotPriceInfo();
  const lastBalanceSnapshot = useLatestBalanceSnapshot();

  if (!longshortGroupQuery.data || !binancePriceQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const infoDict = getLongShortGroupTradeInfo(
    longshortGroupQuery.data,
    binancePriceQuery.data,
  );

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <div>
          <Heading size={"lg"}>
            {longshortGroupQuery.data.group.name.replace("{SYMBOL}_", "")}
          </Heading>
          <Text fontSize={"13px"}>
            Went live:{" "}
            {getDiffToPresentFormatted(
              new Date(longshortGroupQuery.data.group.created_at),
            )}{" "}
            ago
          </Text>
        </div>
      </div>

      <div style={{ marginTop: "8px" }}>
        <ChakraCard>
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Inception date</StatLabel>
                <StatNumber>
                  {new Date(
                    longshortGroupQuery.data.group.created_at,
                  ).toLocaleString("default", {
                    year: "numeric",
                    month: "short",
                    day: "numeric",
                  })}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Universe size</StatLabel>
                <StatNumber>{infoDict.numStrategies}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Trades</StatLabel>
                <StatNumber>{infoDict.totalTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Open positions</StatLabel>
                <StatNumber>{infoDict.openTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Realized profit</StatLabel>
                <StatNumber
                  color={getNumberDisplayColor(
                    infoDict.cumulativeNetResult,
                    COLOR_CONTENT_PRIMARY,
                  )}
                >
                  ${roundNumberFloor(infoDict.cumulativeNetResult, 2)}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Unrealized profit</StatLabel>
                <StatNumber
                  color={getNumberDisplayColor(
                    infoDict.cumulativeUnrealizedProfit,
                    COLOR_CONTENT_PRIMARY,
                  )}
                >
                  ${roundNumberFloor(infoDict.cumulativeUnrealizedProfit, 2)}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Mean allocation</StatLabel>
                <StatNumber>
                  {roundNumberFloor(infoDict.meanAllocation * 100, 2)}%
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Max allocated size</StatLabel>
                <StatNumber>
                  {roundNumberFloor(
                    longshortGroupQuery.data.group.max_leverage_ratio * 100,
                    2,
                  )}
                  %
                </StatNumber>
              </Stat>
            </div>

            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Max simultaneous pos.</StatLabel>
                <StatNumber>
                  {longshortGroupQuery.data.group.max_simultaneous_positions}
                </StatNumber>
              </Stat>
            </div>

            {lastBalanceSnapshot.data && (
              <div>
                <Stat color={COLOR_CONTENT_PRIMARY}>
                  <StatLabel>Long side positions</StatLabel>
                  <StatNumber>
                    {roundNumberFloor(
                      safeDivide(
                        infoDict.longSidePositions,
                        lastBalanceSnapshot.data.value,
                        0,
                      ) * 100,
                      2,
                    )}
                    %
                  </StatNumber>
                </Stat>
              </div>
            )}
            {lastBalanceSnapshot.data && (
              <div>
                <Stat color={COLOR_CONTENT_PRIMARY}>
                  <StatLabel>Short side positions</StatLabel>
                  <StatNumber>
                    {roundNumberFloor(
                      safeDivide(
                        infoDict.shortSidePositions,
                        lastBalanceSnapshot.data.value,
                        0,
                      ) * 100,
                      2,
                    )}
                    %
                  </StatNumber>
                </Stat>
              </div>
            )}
          </div>
        </ChakraCard>
      </div>
    </div>
  );
};
