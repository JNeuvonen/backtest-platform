import { TRADE_DIRECTIONS } from "./strategy_utils";
import { Trade } from "./types";

export const getTradeCurrentProfitPerc = (
  trade: Trade,
  currentPrice: number,
) => {
  try {
    if (trade.direction === TRADE_DIRECTIONS.long) {
      return (currentPrice / trade.open_price - 1) * 100;
    } else {
      return (currentPrice / trade.open_price - 1) * -1 * 100;
    }
  } catch {
    return 0;
  }
};
