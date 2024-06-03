import { BinanceTickerPriceChange } from "./types";

export const SPOT_EXCHANGE_INFO_ENDPOINT =
  "https://api.binance.com/api/v3/exchangeInfo";

export const SPOT_MARKET_INFO_ENDPOINT =
  "https://api.binance.com/api/v3/ticker/price";

export const SPOT_MARKET_PRICE_CHANGE_ENDPOINT =
  "https://api.binance.com/api/v3/ticker/24hr";

export const ASSETS = {
  usdt: "USDT",
  btc: "BTC",
};

export const findSymbolPriceChangeTicker = (
  symbol: string,
  binancePriceChanges: BinanceTickerPriceChange[],
) => {
  for (let i = 0; i < binancePriceChanges.length; ++i) {
    const ticker = binancePriceChanges[i];

    if (ticker.symbol === symbol) {
      return ticker;
    }
  }
  return null;
};
