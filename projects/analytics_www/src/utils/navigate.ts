import { PATHS, PATH_KEYS } from "./app";

export const getStrategyPath = (strategyName: string) => {
  return PATHS.strategy.replace(PATH_KEYS.strategyName, strategyName);
};

export const getLsStrategyPath = (strategyName: string) => {
  return PATHS.lsStrategy.replace(PATH_KEYS.strategyName, strategyName);
};