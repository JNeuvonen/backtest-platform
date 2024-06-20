import {
  Button,
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
  Text,
  useDisclosure,
} from "@chakra-ui/react";
import {
  getDiffToPresentFormatted,
  getNumberDisplayColor,
  getStrategyGroupTradeInfo,
  roundNumberFloor,
} from "src/common_js";
import { useNavigate } from "react-router-dom";
import { ChakraCard } from "src/components/chakra";
import { usePathParams } from "src/hooks";
import {
  useBalanceSnapshotsQuery,
  useBinanceSpotPriceInfo,
  useStrategyGroupQuery,
} from "src/http/queries";
import { BUTTON_VARIANTS, COLOR_CONTENT_PRIMARY } from "src/theme";
import { getStrategySymbolsPath } from "src/utils";
import { ConfirmModal } from "src/components/ConfirmModal";
import { disableAndCloseStratGroup } from "src/http";
import { toast } from "react-toastify";

export const StrategyPage = () => {
  const { strategyName } = usePathParams<{ strategyName: string }>();
  const strategyGroupQuery = useStrategyGroupQuery(strategyName);
  const binancePriceQuery = useBinanceSpotPriceInfo();
  const balanceSnapShots = useBalanceSnapshotsQuery();
  const navigate = useNavigate();
  const disableAndCloseConfirmModal = useDisclosure();

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

  const disableAndClosePositions = async () => {
    const res = await disableAndCloseStratGroup(
      strategyGroupQuery.data.strategy_group.id,
    );

    if (res.success) {
      toast.success("Disabled strategy", { theme: "dark" });
    } else {
      toast.error("Failed to disable strategy", { theme: "dark" });
    }
  };

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
      <ConfirmModal
        {...disableAndCloseConfirmModal}
        onConfirm={disableAndClosePositions}
        title={"Disable strategy"}
        message={
          "Are you sure you want to disable this strategy and close all positions?"
        }
      />
      <div style={{ display: "flex", justifyContent: "space-between" }}>
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

            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Mean allocation</StatLabel>
                <StatNumber>
                  {roundNumberFloor(tradeInfoDict.meanAllocation, 2)}%
                </StatNumber>
              </Stat>
            </div>

            {lastBalanceSnapshot && (
              <div>
                <Stat color={COLOR_CONTENT_PRIMARY}>
                  <StatLabel>Value at risk</StatLabel>
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

      <div
        style={{
          marginTop: "16px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <Button
          variant={BUTTON_VARIANTS.dangerNoFill}
          onClick={() => disableAndCloseConfirmModal.onOpen()}
        >
          Disable and close positions
        </Button>
        <Button
          variant={BUTTON_VARIANTS.nofill}
          onClick={() => navigate(getStrategySymbolsPath(strategyName))}
        >
          Tickers
        </Button>
      </div>
    </div>
  );
};
