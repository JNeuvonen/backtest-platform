import React from "react";
import { ChakraCard } from "../../../../components/chakra/Card";
import { Heading, Stat, StatLabel, StatNumber } from "@chakra-ui/react";
import { COLOR_CONTENT_PRIMARY } from "../../../../utils/colors";
import { roundNumberDropRemaining } from "../../../../utils/number";
import { BacktestObject } from "../../../../clients/queries/response-types";
import { formatSecondsIntoTime } from "../../../../utils/date";

interface Props {
  backtest: BacktestObject;
  dateRange: string;
}

export const BacktestSummaryCard = (props: Props) => {
  const { backtest, dateRange } = props;
  return (
    <div style={{ marginTop: "16px" }}>
      <ChakraCard heading={<Heading size="md">Summary: {dateRange}</Heading>}>
        <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Risk adjusted CAGR</StatLabel>
              <StatNumber>
                {backtest.risk_adjusted_return
                  ? String(
                      roundNumberDropRemaining(
                        backtest.risk_adjusted_return * 100,
                        2
                      )
                    ) + "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Actual CAGR</StatLabel>
              <StatNumber>
                {backtest.cagr
                  ? String(roundNumberDropRemaining(backtest.cagr * 100, 2)) +
                    "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Buy and hold CAGR</StatLabel>
              <StatNumber>
                {backtest.buy_and_hold_cagr
                  ? String(
                      roundNumberDropRemaining(
                        backtest.buy_and_hold_cagr * 100,
                        2
                      )
                    ) + "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Time exposure</StatLabel>
              <StatNumber>
                {backtest.market_exposure_time
                  ? String(
                      roundNumberDropRemaining(
                        backtest.market_exposure_time * 100,
                        2
                      )
                    ) + "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Max drawdown</StatLabel>
              <StatNumber>
                {backtest.max_drawdown_perc
                  ? String(
                      roundNumberDropRemaining(backtest.max_drawdown_perc, 2)
                    ) + "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>

          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Profit factor</StatLabel>
              <StatNumber>
                {backtest.profit_factor
                  ? String(roundNumberDropRemaining(backtest.profit_factor, 2))
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Total trade count</StatLabel>
              <StatNumber>
                {backtest.profit_factor
                  ? String(roundNumberDropRemaining(backtest.trade_count, 2))
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>

          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Winning trades</StatLabel>
              <StatNumber>
                {backtest.share_of_winning_trades_perc
                  ? String(
                      roundNumberDropRemaining(
                        backtest.share_of_winning_trades_perc,
                        2
                      )
                    ) + "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Losing trades</StatLabel>
              <StatNumber>
                {backtest.share_of_losing_trades_perc
                  ? String(
                      roundNumberDropRemaining(
                        backtest.share_of_losing_trades_perc,
                        2
                      )
                    ) + "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>

          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Trade mean return</StatLabel>
              <StatNumber>
                {backtest.mean_return_perc
                  ? String(
                      roundNumberDropRemaining(backtest.mean_return_perc, 2)
                    ) + "%"
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>

          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>Mean pos. time</StatLabel>
              <StatNumber>
                {backtest.mean_hold_time_sec
                  ? String(formatSecondsIntoTime(backtest.mean_hold_time_sec))
                  : "N/A"}
              </StatNumber>
            </Stat>
          </div>

          {backtest.is_long_short_strategy && (
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Asset universe size</StatLabel>
                <StatNumber>
                  {backtest.asset_universe_size
                    ? String(backtest.asset_universe_size)
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>
          )}

          {backtest.is_long_short_strategy && (
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Long profit factor</StatLabel>
                <StatNumber>
                  {backtest.long_side_profit_factor
                    ? String(
                        roundNumberDropRemaining(
                          backtest.long_side_profit_factor,
                          2
                        )
                      )
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>
          )}
          {backtest.is_long_short_strategy && (
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Short profit factor</StatLabel>
                <StatNumber>
                  {backtest.short_side_profit_factor
                    ? String(
                        roundNumberDropRemaining(
                          backtest.short_side_profit_factor,
                          2
                        )
                      )
                    : "N/A"}
                </StatNumber>
              </Stat>
            </div>
          )}
        </div>
      </ChakraCard>
    </div>
  );
};
