import { safeDivide } from "./math";
import { BinanceSymbolPrice, Strategy, StrategyGroup, Trade } from "./types";

export const findCurrentPrice = (
  symbol: string,
  latestPrices: BinanceSymbolPrice[],
) => {
  for (let i = 0; i < latestPrices.length; ++i) {
    if (latestPrices[i].symbol === symbol) {
      return Number(latestPrices[i].price);
    }
  }
  return null;
};

export const getStrategyGroupTradeInfo = (
  strategyGroup: StrategyGroup,
  strategies: Strategy[],
  trades: Trade[],
  latestPrices: BinanceSymbolPrice[],
) => {
  let openTrades = 0;
  let closedTrades = 0;
  let cumulativeNetResult = 0;
  let cumulativePercResult = 0;
  let numStrategies = 0;
  let cumulativeUnrealizedProfit = 0;
  let isLongStrategy = true;
  let positionSize = 0;

  const stratIdSet = new Set();

  strategies.forEach((item) => {
    if (item.strategy_group_id === strategyGroup.id) {
      if (item.is_short_selling_strategy) {
        isLongStrategy = false;
      }
      stratIdSet.add(item.id);
      numStrategies += 1;
    }
  });

  trades.forEach((item) => {
    if (stratIdSet.has(item.strategy_id)) {
      if (item.close_price && item.net_result && item.percent_result) {
        closedTrades += 1;
        cumulativeNetResult += item.net_result;
        cumulativePercResult += item.percent_result;
      } else {
        openTrades += 1;
        const latestPrice = findCurrentPrice(item.symbol, latestPrices);

        if (isLongStrategy && latestPrice) {
          const unrealizedProfit =
            (latestPrice - item.open_price) * item.quantity;
          cumulativeUnrealizedProfit += unrealizedProfit;
        }

        if (!isLongStrategy && latestPrice) {
          const unrealizedProfit =
            (item.open_price - latestPrice) * item.quantity;

          cumulativeUnrealizedProfit += unrealizedProfit;
        }

        if (latestPrice) {
          positionSize += item.quantity * latestPrice;
        }
      }
    }
  });

  return {
    openTrades,
    closedTrades,
    totalTrades: openTrades + closedTrades,
    cumulativePercResult,
    cumulativeNetResult,
    meanTradeResult: safeDivide(cumulativePercResult, closedTrades, 0),
    numStrategies,
    cumulativeUnrealizedProfit,
    positionSize,
  };
};
