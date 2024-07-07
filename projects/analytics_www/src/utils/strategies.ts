import { StrategyGroup } from "src/common_js";
import { StrategiesTableFilters } from "src/pages";

export const filterStrategiesRows = (
  strategyGroup: StrategyGroup,
  filters: StrategiesTableFilters,
) => {
  let enabledStatePass = false;
  let strategyDirectionPass = false;

  if (filters.enabledStrategies && filters.disabledStrategies) {
    enabledStatePass = true;
  } else if (filters.enabledStrategies && !strategyGroup.is_disabled) {
    enabledStatePass = true;
  } else if (filters.disabledStrategies && strategyGroup.is_disabled) {
    enabledStatePass = true;
  }

  if (filters.longStrategies && filters.shortStrategies) {
    strategyDirectionPass = true;
  }
  if (filters.longStrategies && !strategyGroup.is_short_selling_strategy) {
    strategyDirectionPass = true;
  } else if (
    filters.shortStrategies &&
    strategyGroup.is_short_selling_strategy
  ) {
    strategyDirectionPass = true;
  }

  return enabledStatePass && strategyDirectionPass;
};

interface FormattedStrategyRow {
  size: number;
  numTrades: number;
  netResult: number;
  openTrades: number;
  meanTradeResultPerc: number;
  closedTrades: number;
  cumulativeUnrealizedProfit: number;
  valueInOpenPositions: number;
  cumulativePosHoldTime: number;
  meanPosHoldTimeMs: number;
  isLongStrategy: boolean;
  name: string;
}

export const filterFormattedStrategiesRows = (
  row: FormattedStrategyRow,
  filters: StrategiesTableFilters,
) => {
  let positionStatePass = false;

  if (filters.strategiesInPosition && filters.strategiesOutOfPosition) {
    positionStatePass = true;
  } else if (filters.strategiesInPosition && row.openTrades > 0) {
    positionStatePass = true;
  } else if (filters.strategiesOutOfPosition && row.openTrades === 0) {
    positionStatePass = true;
  }
  return positionStatePass;
};
