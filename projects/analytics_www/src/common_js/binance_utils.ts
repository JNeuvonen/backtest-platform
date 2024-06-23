import { BinanceTickerPriceChange, Trade } from "./types";

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

export function inferAssets(symbol: string): {
  baseAsset: string;
  quoteAsset: string;
} {
  let baseAsset: string;
  let quoteAsset: string;

  if (symbol.endsWith("USDT")) {
    quoteAsset = "USDT";
    baseAsset = symbol.slice(0, symbol.length - 4);
  } else if (symbol.endsWith("BTC")) {
    quoteAsset = "BTC";
    baseAsset = symbol.slice(0, symbol.length - 3);
  } else if (symbol.endsWith("ETH")) {
    quoteAsset = "ETH";
    baseAsset = symbol.slice(0, symbol.length - 3);
  } else if (symbol.endsWith("BNB")) {
    quoteAsset = "BNB";
    baseAsset = symbol.slice(0, symbol.length - 3);
  } else {
    baseAsset = "";
    quoteAsset = "";
  }

  return { baseAsset, quoteAsset };
}

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

export const findNumOpenPositions = (baseAsset: string, trades: Trade[]) => {
  let numOpenTrades = 0;
  trades.forEach((item) => {
    const assets = inferAssets(item.symbol);
    if (assets["baseAsset"] === baseAsset) {
      numOpenTrades += 1;
    }
  });
  return numOpenTrades;
};
