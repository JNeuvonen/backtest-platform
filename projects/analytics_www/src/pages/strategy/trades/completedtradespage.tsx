import {
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
} from "@chakra-ui/react";
import {
  getNumberDisplayColor,
  roundNumberFloor,
  safeDivide,
} from "src/common_js";
import { ChakraCard, ClosedTradesTable } from "src/components";
import { OpenTradesTable } from "src/components/OpenTradesTable";
import { usePathParams } from "src/hooks";
import { useStrategyGroupQuery } from "src/http";
import { COLOR_CONTENT_PRIMARY } from "src/theme";

export const StratGroupCompletedTradesPage = () => {
  const { strategyName } = usePathParams<{ strategyName: string }>();
  const strategyGroupQuery = useStrategyGroupQuery(strategyName);

  if (!strategyGroupQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const getInfoDict = () => {
    let realizedProfit = 0;
    let cumulativeRealizedProfitPerc = 0;
    let numTrades = 0;

    strategyGroupQuery.data.trades.forEach((item) => {
      if (item.close_price) {
        realizedProfit += item.net_result;
        cumulativeRealizedProfitPerc += item.percent_result;
        numTrades += 1;
      }
    });

    return {
      realizedProfit,
      numTrades,
      meanResultPerc: safeDivide(cumulativeRealizedProfitPerc, numTrades, 0),
    };
  };

  const infoDict = getInfoDict();

  return (
    <div>
      <div>
        <Heading size={"lg"}>Completed trades for {strategyName}</Heading>
      </div>

      <ChakraCard>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            flexWrap: "wrap",
          }}
        >
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
              <StatLabel>Direction</StatLabel>
              <StatNumber>
                {strategyGroupQuery.data.strategy_group
                  .is_short_selling_strategy
                  ? "Short"
                  : "Long"}
              </StatNumber>
            </Stat>
          </div>

          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Realized profit</StatLabel>
              <StatNumber
                color={getNumberDisplayColor(
                  infoDict.realizedProfit,
                  COLOR_CONTENT_PRIMARY,
                )}
              >
                ${roundNumberFloor(infoDict.realizedProfit, 2)}
              </StatNumber>
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Num trades</StatLabel>
              <StatNumber>{infoDict.numTrades}</StatNumber>
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Mean trade (%)</StatLabel>
              <StatNumber>
                {roundNumberFloor(infoDict.meanResultPerc, 2)}%
              </StatNumber>
            </Stat>
          </div>
        </div>
      </ChakraCard>
      <div style={{ marginTop: "16px" }}>
        <Heading size={"md"}>Closed trades</Heading>
        <ClosedTradesTable
          trades={strategyGroupQuery.data.trades.filter(
            (item) => item.close_price !== null,
          )}
        />
      </div>
      <div style={{ marginTop: "16px" }}>
        <Heading size={"md"}>Open positions</Heading>
        <OpenTradesTable
          trades={strategyGroupQuery.data.trades.filter(
            (item) => item.close_price === null,
          )}
        />
      </div>
    </div>
  );
};
