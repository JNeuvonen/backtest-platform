import { LongShortPair } from "./types";

export const findLongShortPair = (tickerId: number, pairs: LongShortPair[]) => {
  for (let i = 0; i < pairs.length; ++i) {
    const pair = pairs[i];

    if (pair.buy_ticker_id === tickerId || tickerId === pair.sell_ticker_id) {
      return pair;
    }
  }
  return null;
};
