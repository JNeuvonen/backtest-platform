export const PATH_KEYS = {
  strategyName: ":strategyName",
  id: ":id",
};

const STRATEGIES_PATH = "/strategies";
const LS_STRATEGY_PATH = `/ls-strategy/${PATH_KEYS.strategyName}`;
const PROFILE_PATH = "/profile";
const STRATEGY_PATH = `/strategy/${PATH_KEYS.strategyName}`;
const TRADE_PATH = "/trade";
const VIEW_TRADE_PATH = TRADE_PATH + `/${PATH_KEYS.id}`;
const STRATEGY_SYMBOLS_PATH = STRATEGY_PATH + `/strategies`;

export const PATHS = {
  root: "/",
  dashboard: "/",
  strategies: STRATEGIES_PATH,
  profile: PROFILE_PATH,
  strategy: STRATEGY_PATH,
  lsStrategy: LS_STRATEGY_PATH,
  strategySymbols: STRATEGY_SYMBOLS_PATH,
  viewTradePath: VIEW_TRADE_PATH,
};

export const SIDENAV_DEFAULT_WIDTH = 140;
export const MOBILE_WIDTH_CUTOFF = 1000;
