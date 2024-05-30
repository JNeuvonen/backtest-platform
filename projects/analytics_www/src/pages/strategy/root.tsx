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
  getNumberDisplayColor,
  getStrategyGroupTradeInfo,
  roundNumberFloor,
} from "common_js";
import { ChakraCard } from "src/components/chakra";
import { usePathParams } from "src/hooks";
import {
  useBalanceSnapshotsQuery,
  useBinanceSpotPriceInfo,
  useStrategyGroupQuery,
} from "src/http/queries";
import { COLOR_CONTENT_PRIMARY } from "src/theme";

export const StrategyPage = () => {
  const { strategyName } = usePathParams<{ strategyName: string }>();
  const strategyGroupQuery = useStrategyGroupQuery(strategyName);
  const binancePriceQuery = useBinanceSpotPriceInfo();
  const balanceSnapShots = useBalanceSnapshotsQuery();

  if (
    strategyGroupQuery.isLoading ||
    !strategyGroupQuery.data ||
    !binancePriceQuery.data ||
    !balanceSnapShots.data ||
    balanceSnapShots.isLoading
  ) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const tradeInfoDict = getStrategyGroupTradeInfo(
    strategyGroupQuery.data.strategy_group,
    strategyGroupQuery.data.strategies,
    strategyGroupQuery.data.trades,
    binancePriceQuery.data,
  );

  const lastBalanceSnapshot =
    balanceSnapShots.data[balanceSnapShots.data.length - 1];

  return (
    <div>
      <div>
        <Heading size={"lg"}>
          {strategyGroupQuery.data.strategy_group.name}
        </Heading>
        <Text fontSize={"13px"}>
          Went live:{" "}
          {getDiffToPresentFormatted(
            new Date(strategyGroupQuery.data.strategy_group.created_at),
          )}{" "}
          ago
        </Text>
      </div>
      <div style={{ marginTop: "8px" }}>
        <ChakraCard>
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Inception date</StatLabel>
                <StatNumber>
                  {new Date(
                    strategyGroupQuery.data.strategy_group.created_at,
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
                <StatNumber>{tradeInfoDict.numStrategies}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Trades</StatLabel>
                <StatNumber>{tradeInfoDict.totalTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Open trades</StatLabel>
                <StatNumber>{tradeInfoDict.openTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Closed trades</StatLabel>
                <StatNumber>{tradeInfoDict.closedTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Realized profit</StatLabel>
                <StatNumber
                  color={getNumberDisplayColor(
                    tradeInfoDict.cumulativeNetResult,
                    COLOR_CONTENT_PRIMARY,
                  )}
                >
                  ${roundNumberFloor(tradeInfoDict.cumulativeNetResult, 2)}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Unrealized profit</StatLabel>
                <StatNumber
                  color={getNumberDisplayColor(
                    tradeInfoDict.cumulativeUnrealizedProfit,
                    COLOR_CONTENT_PRIMARY,
                  )}
                >
                  $
                  {roundNumberFloor(
                    tradeInfoDict.cumulativeUnrealizedProfit,
                    2,
                  )}
                </StatNumber>
              </Stat>
            </div>

            {lastBalanceSnapshot && (
              <div>
                <Stat color={COLOR_CONTENT_PRIMARY}>
                  <StatLabel>Position size</StatLabel>
                  <StatNumber>
                    {roundNumberFloor(
                      (tradeInfoDict.positionSize / lastBalanceSnapshot.value) *
                        100,
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