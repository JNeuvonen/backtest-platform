import { Heading, Stat, StatLabel, StatNumber } from "@chakra-ui/react";
import {
  BalanceSnapshot,
  getRateOfChangePerc,
  roundNumberFloor,
} from "src/common_js";
import { COLOR_CONTENT_PRIMARY } from "src/theme";
import { ChakraCard } from "./chakra";
import ChakraStatHelpText from "./chakra/StatHelpText";

interface InfoCardProps {
  heading: React.ReactNode;
  lastTick: BalanceSnapshot;
  comparisonTick: BalanceSnapshot;
  showOnlyNav?: boolean;
  showOnlyDiff?: boolean;
}

const BalanceInfoCard: React.FC<InfoCardProps> = ({
  heading,
  lastTick,
  comparisonTick,
  showOnlyNav = false,
  showOnlyDiff = false,
}) => {
  return (
    <ChakraCard heading={<Heading size="md">{heading}</Heading>}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "16px",
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>NAV</StatLabel>
              {!showOnlyDiff && (
                <StatNumber>${roundNumberFloor(lastTick.value, 2)}</StatNumber>
              )}
              <ChakraStatHelpText
                num={roundNumberFloor(
                  getRateOfChangePerc(lastTick.value, comparisonTick?.value),
                  2,
                )}
              />
            </Stat>
          </div>
          <div>
            <Stat color={COLOR_CONTENT_PRIMARY}>
              <StatLabel>BTC price</StatLabel>
              {!showOnlyDiff && (
                <StatNumber>
                  ${roundNumberFloor(lastTick.btc_price, 2)}
                </StatNumber>
              )}
              <ChakraStatHelpText
                num={roundNumberFloor(
                  getRateOfChangePerc(
                    lastTick.btc_price,
                    comparisonTick?.btc_price,
                  ),
                  2,
                )}
              />
            </Stat>
          </div>
        </div>
        {!showOnlyNav && (
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
                <StatLabel>Debt</StatLabel>
                <StatNumber>${roundNumberFloor(lastTick.debt, 2)}</StatNumber>
                <ChakraStatHelpText
                  num={roundNumberFloor(
                    getRateOfChangePerc(lastTick.debt, comparisonTick.debt),
                    2,
                  )}
                />
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Longs</StatLabel>
                <StatNumber>
                  ${roundNumberFloor(lastTick.long_assets_value, 2)}
                </StatNumber>
                <ChakraStatHelpText
                  num={roundNumberFloor(
                    getRateOfChangePerc(
                      lastTick.long_assets_value,
                      comparisonTick.long_assets_value,
                    ),
                    2,
                  )}
                />
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Margin level</StatLabel>
                <StatNumber>
                  {roundNumberFloor(lastTick.margin_level, 2)}
                </StatNumber>
                <ChakraStatHelpText
                  num={roundNumberFloor(
                    getRateOfChangePerc(
                      lastTick.margin_level,
                      comparisonTick.margin_level,
                    ),
                    2,
                  )}
                />
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Long positions</StatLabel>
                <StatNumber>{lastTick.num_long_positions}</StatNumber>
                <ChakraStatHelpText
                  num={
                    lastTick.num_long_positions -
                    comparisonTick.num_long_positions
                  }
                  percentage={false}
                />
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Short positions</StatLabel>
                <StatNumber>{lastTick.num_short_positions}</StatNumber>
                <ChakraStatHelpText
                  num={
                    lastTick.num_short_positions -
                    comparisonTick.num_short_positions
                  }
                  percentage={false}
                />
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Active pair trades</StatLabel>
                <StatNumber>{lastTick.num_ls_positions}</StatNumber>
                <ChakraStatHelpText
                  num={
                    lastTick.num_ls_positions - comparisonTick.num_ls_positions
                  }
                  percentage={false}
                />
              </Stat>
            </div>
          </div>
        )}
      </div>
    </ChakraCard>
  );
};

export default BalanceInfoCard;
