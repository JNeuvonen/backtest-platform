export const PATH_KEYS = {
  strategyName: ":strategyName",
  id: ":id",
};

const STRATEGIES_PATH = "/strategies";
const REPROD_LIVE_TRADES_PATH = `/strategies/reproduce/${PATH_KEYS.strategyName}`;
const ASSETS_PATH = "/assets";
const LS_STRATEGY_PATH = `/ls-strategy/${PATH_KEYS.strategyName}`;
const PROFILE_PATH = "/profile";
const STRATEGY_PATH = `/strategy/${PATH_KEYS.strategyName}`;
const TRADE_PATH = "/trade";
const VIEW_TRADE_PATH = TRADE_PATH + `/${PATH_KEYS.id}`;
const STRATEGY_SYMBOLS_PATH = STRATEGY_PATH + `/strategies`;
const LS_TICKERS_PATH = LS_STRATEGY_PATH + "/tickers";
const STRATEGY_GROUP_COMPLETED_TRADES = `/strategy/${PATH_KEYS.strategyName}/completed-trades`;

export const PATHS = {
  root: "/",
  dashboard: "/",
  strategies: STRATEGIES_PATH,
  assets: ASSETS_PATH,
  profile: PROFILE_PATH,
  strategy: STRATEGY_PATH,
  lsStrategy: LS_STRATEGY_PATH,
  lsTickers: LS_TICKERS_PATH,
  strategySymbols: STRATEGY_SYMBOLS_PATH,
  viewTradePath: VIEW_TRADE_PATH,
  reproduceLiveTradesPath: REPROD_LIVE_TRADES_PATH,
  strategyGroupCompletedTrades: STRATEGY_GROUP_COMPLETED_TRADES,
};

export const SIDENAV_DEFAULT_WIDTH = 140;
export const TOP_BAR_HEIGHT = 45;
export const MOBILE_WIDTH_CUTOFF = 1000;
