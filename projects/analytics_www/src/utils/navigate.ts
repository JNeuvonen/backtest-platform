import { PATHS, PATH_KEYS } from "./app";

export const getStrategyPath = (strategyName: string) => {
  return PATHS.strategy.replace(PATH_KEYS.strategyName, strategyName);
};

export const getLsStrategyPath = (strategyName: string) => {
  return PATHS.lsStrategy.replace(PATH_KEYS.strategyName, strategyName);
};

export const getStrategySymbolsPath = (strategyName: string) => {
  return PATHS.strategySymbols.replace(PATH_KEYS.strategyName, strategyName);
};

export const getLongShortTickersPath = (strategyName: string) => {
  return PATHS.lsTickers.replace(PATH_KEYS.strategyName, strategyName);
};

export const getTradeViewPath = (id: number) => {
  return PATHS.viewTradePath.replace(PATH_KEYS.id, String(id));
};
