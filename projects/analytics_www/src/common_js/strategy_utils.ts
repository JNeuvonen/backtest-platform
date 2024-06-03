import { safeDivide } from "./math";
import {
  BinanceSymbolPrice,
  LongShortGroupResponse,
  Strategy,
  StrategyGroup,
  Trade,
} from "./types";

export const TRADE_DIRECTIONS = {
  long: "LONG",
  short: "SHORT",
};

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
  let cumulativeAllocation = 0;

  const stratIdSet = new Set();

  strategies.forEach((item) => {
    if (item.strategy_group_id === strategyGroup.id) {
      if (item.is_short_selling_strategy) {
        isLongStrategy = false;
      }
      stratIdSet.add(item.id);
      numStrategies += 1;

      if (item.allocated_size_perc) {
        cumulativeAllocation += item.allocated_size_perc;
      }
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
    meanAllocation: safeDivide(cumulativeAllocation, numStrategies, 0),
  };
};

export const getLongShortGroupTradeInfo = (
  longShortGroupRes: LongShortGroupResponse,
  latestPrices: BinanceSymbolPrice[],
) => {
  let openTrades = 0;
  let cumulativeNetResult = 0;
  let cumulativePercResult = 0;
  let cumulativeHoldTimeMs = 0;
  let cumulativeUnrealizedProfit = 0;
  let positionSize = 0;
  let longSidePositions = 0;
  let shortSidePositions = 0;

  longShortGroupRes.completed_trades.forEach((item) => {
    cumulativeNetResult += item.net_result;
    cumulativePercResult += item.percent_result;
    cumulativeHoldTimeMs += item.close_time_ms - item.open_time_ms;
  });

  const openTradesArr = longShortGroupRes.trades;
  const numClosedTrades = longShortGroupRes.completed_trades.length;

  openTradesArr.forEach((item) => {
    if (!item.percent_result) {
      const latestPrice = findCurrentPrice(item.symbol, latestPrices);
      openTrades += 1;

      if (!latestPrice) return;

      if (item.direction === TRADE_DIRECTIONS.long) {
        const unrealizedProfit =
          (latestPrice - item.open_price) * item.quantity;
        cumulativeUnrealizedProfit += unrealizedProfit;
        longSidePositions += latestPrice * item.quantity;
      } else {
        const unrealizedProfit =
          (item.open_price - latestPrice) * item.quantity;
        cumulativeUnrealizedProfit += unrealizedProfit;
        shortSidePositions += latestPrice * item.quantity;
      }
    }
  });

  return {
    openTrades,
    closedTrades: numClosedTrades,
    totalTrades: openTrades + numClosedTrades,
    cumulativePercResult,
    cumulativeNetResult,
    meanTradeResult: safeDivide(cumulativePercResult, numClosedTrades, 0),
    numStrategies: longShortGroupRes.tickers.length,
    cumulativeUnrealizedProfit,
    positionSize,
    meanPositionTimeMs: safeDivide(cumulativeHoldTimeMs, numClosedTrades, 0),
    longSidePositions,
    shortSidePositions,
    meanAllocation: safeDivide(
      longShortGroupRes.group.max_leverage_ratio || 0,
      longShortGroupRes.group.max_simultaneous_positions || 0,
      0,
    ),
  };
};
