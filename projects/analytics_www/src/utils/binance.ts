export const BINANCE_VIEW_CHART_URI =
  "https://www.binance.com/en/trade/{BASE}_{QUOTE}?type=cross";

export const getViewBinanceChartUri = (
  baseAsset: string,
  quoteAsset: string,
) => {
  return BINANCE_VIEW_CHART_URI.replace("{BASE}", baseAsset).replace(
    "{QUOTE}",
    quoteAsset,
  );
};
