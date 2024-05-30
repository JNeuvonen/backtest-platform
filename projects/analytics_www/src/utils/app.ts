export const PATH_KEYS = {
  strategyName: ":strategyName",
};

const STRATEGIES_PATH = "/strategies";
const LS_STRATEGY_PATH = `/ls-strategy/${PATH_KEYS.strategyName}`;
const PROFILE_PATH = "/profile";
const STRATEGY_PATH = `/strategy/${PATH_KEYS.strategyName}`;

export const PATHS = {
  root: "/",
  dashboard: "/",
  strategies: STRATEGIES_PATH,
  profile: PROFILE_PATH,
  strategy: STRATEGY_PATH,
  lsStrategy: LS_STRATEGY_PATH,
};

export const SIDENAV_DEFAULT_WIDTH = 140;
export const MOBILE_WIDTH_CUTOFF = 1000;
