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
    throw new Error("Unsupported quote asset.");
  }

  return { baseAsset, quoteAsset };
}

export const MINUTE_IN_MS = 60000;
export const HOUR_IN_MS = MINUTE_IN_MS * 60;
export const DAY_IN_MS = HOUR_IN_MS * 24;
export const SPOT_EXCHANGE_INFO_ENDPOINT =
  "https://api.binance.com/api/v3/exchangeInfo";

export const getIntervalLengthInMs = (interval: string): number => {
  const intervals = {
    "1m": MINUTE_IN_MS,
    "3m": MINUTE_IN_MS * 3,
    "5m": MINUTE_IN_MS * 5,
    "15m": MINUTE_IN_MS * 15,
    "30m": MINUTE_IN_MS * 30,
    "1h": HOUR_IN_MS,
    "2h": HOUR_IN_MS * 2,
    "4h": HOUR_IN_MS * 4,
    "6h": HOUR_IN_MS * 6,
    "8h": HOUR_IN_MS * 8,
    "12h": HOUR_IN_MS * 12,
    "1d": DAY_IN_MS,
    "3d": DAY_IN_MS * 3,
    "1w": DAY_IN_MS * 7,
    "1M": DAY_IN_MS * 30,
  };

  return intervals[interval] || 0;
};

export async function getTradeQuantityPrecision(
  symbol: string
): Promise<number> {
  const response = await fetch(SPOT_EXCHANGE_INFO_ENDPOINT);
  const data = await response.json();

  for (const item of data.symbols) {
    if (item.symbol === symbol) {
      for (const filter of item.filters) {
        if (filter.filterType === "LOT_SIZE") {
          const tradeQuantityPrecision = filter.minQty.includes(".")
            ? filter.minQty.split(".")[1].replace(/0+$/, "").length
            : 0;
          return Number(tradeQuantityPrecision);
        }
      }
    }
  }
  throw new Error("Symbol doesn't exist");
}
