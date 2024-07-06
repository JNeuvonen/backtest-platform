import { off } from "process";
import { FetchBulkBacktests, Trade } from "../clients/queries/response-types";
import { binarySearch } from "./algo";
import { getKeysCount } from "./object";
import { DisplayPairsItem } from "../pages/mass-backtest/backtest";

const KLINE_OPEN_TIME_KEY = "kline_open_time";
const EQUITY_KEY = "equity";
const BACKTEST_START_BALANCE = 10000;
export const COMBINED_STRATEGY_DATA_KEY = "combined_equity";

export const getMassSimFindTicks = (
  bulkFetchBacktest: FetchBulkBacktests,
  klineOpenTime: number
) => {
  const ret = {};

  bulkFetchBacktest.equity_curves.forEach((item) => {
    for (const [key, equityCurveTicks] of Object.entries(item)) {
      const item = binarySearch(
        equityCurveTicks,
        klineOpenTime,
        (tick) => tick.kline_open_time
      );

      ret[bulkFetchBacktest.id_to_dataset_name_map[key]] =
        item?.portfolio_worth || null;
    }
  });
  return ret;
};

export const convertToReturn = (
  currItem: object,
  prevItem: object,
  key: string
) => {
  const currVal = currItem[key];
  const prevVal = prevItem[key];

  if (prevVal === 0) {
    return 1;
  }

  const value = currVal / prevVal;

  if (isNaN(value) || !isFinite(value)) {
    return 1;
  }

  return value;
};

export const getEquityCurveStatistics = (
  bulkFetchBacktest: FetchBulkBacktests,
  returns: object[],
  displayPairsArr: DisplayPairsItem[]
) => {
  const endBalances = {};

  bulkFetchBacktest.equity_curves.forEach((item) => {
    for (const [key, equityCurveTicks] of Object.entries(item)) {
      if (key === KLINE_OPEN_TIME_KEY) {
        continue;
      }
      for (let i = equityCurveTicks.length - 1; i >= 0; --i) {
        const tick = equityCurveTicks[i];

        if (tick.portfolio_worth) {
          endBalances[bulkFetchBacktest.id_to_dataset_name_map[key]] =
            tick.portfolio_worth;
          break;
        }
      }
    }
  });

  let multiStratCurrBalance = BACKTEST_START_BALANCE;
  const multiStratBalanceTicks: object[] = [];

  const totalStrats = displayPairsArr.filter((item) => item.display).length;

  const totalReturnsByStrat = {};

  for (const [key] of Object.entries(endBalances)) {
    totalReturnsByStrat[key] = 1;
  }

  for (let i = 0; i < returns.length; ++i) {
    const returnsTick = returns[i];

    const tick = {};
    const roundReturns = {};

    let idx = 0;

    for (const [key, value] of Object.entries(returnsTick)) {
      if (key === KLINE_OPEN_TIME_KEY) {
        tick[KLINE_OPEN_TIME_KEY] = value;
        continue;
      }

      const filteredItem = displayPairsArr.filter(
        (item) => item.datasetSymbol === key
      );

      if (filteredItem.length > 0 && !filteredItem[0].display) {
        continue;
      }

      let coeff = value === 0 ? 1 : value;
      roundReturns[key] = (multiStratCurrBalance / totalStrats) * coeff;
      totalReturnsByStrat[key] = totalReturnsByStrat[key] * coeff;

      idx += 1;
    }

    let tickBalance = 0;

    for (const [_, value] of Object.entries(roundReturns)) {
      tickBalance += value as number;
    }

    multiStratCurrBalance = tickBalance;
    tick[EQUITY_KEY] = multiStratCurrBalance;
    multiStratBalanceTicks.push(tick);
  }

  return {
    endBalances: endBalances,
    multiStrategyBalanceTicks: multiStratBalanceTicks,
    multiStrategyReturnsCurve: convertBalanceTicksToEqCurve(
      multiStratBalanceTicks
    ),
    ...getNumWinningAndLosingStrats(totalReturnsByStrat),
    totalReturnsByStrat,
  };
};

const convertBalanceTicksToReturnTicks = (balanceTicks: object[]) => {
  const returnsCoEff = [] as object[];

  for (let i = 1; i < balanceTicks.length; ++i) {
    const prevItem = balanceTicks[i - 1];
    const currItem = balanceTicks[i];

    const tick = {
      kline_open_time: currItem[KLINE_OPEN_TIME_KEY],
    };

    for (const [key, _] of Object.entries(currItem)) {
      if (key === KLINE_OPEN_TIME_KEY) {
        continue;
      }
      const value = convertToReturn(currItem, prevItem, key);
      tick[key] = value;
    }
    returnsCoEff.push(tick);
  }
  return returnsCoEff;
};

const convertBalanceTicksToEqCurve = (balanceTicks: object[]) => {
  const returnsCoEff = [] as object[];

  for (let i = 1; i < balanceTicks.length; ++i) {
    const prevItem = balanceTicks[i - 1];
    const currItem = balanceTicks[i];

    const tick = {
      kline_open_time: currItem[KLINE_OPEN_TIME_KEY],
    };

    for (const [key, _] of Object.entries(currItem)) {
      if (key === KLINE_OPEN_TIME_KEY) {
        continue;
      }
      const value = convertToReturn(currItem, prevItem, key);
      tick[key] = value;
    }
    returnsCoEff.push(tick);
  }

  const retArr = [] as object[];
  const helper = {};

  for (let i = 0; i < returnsCoEff.length; ++i) {
    const currItem = returnsCoEff[i];
    if (i === 0) {
      for (const [key] of Object.entries(currItem)) {
        if (key === KLINE_OPEN_TIME_KEY) {
          continue;
        }
        helper[key] = 1;
      }
      continue;
    }

    const tick = {
      kline_open_time: currItem[KLINE_OPEN_TIME_KEY],
    };

    for (const [key, value] of Object.entries(currItem)) {
      if (key === KLINE_OPEN_TIME_KEY) {
        continue;
      }
      if (value === undefined || value === null || isNaN(value) || value == 0) {
        tick[key] = (helper[key] - 1) * 100;
        continue;
      }
      helper[key] = helper[key] * value;
      tick[key] = (helper[key] - 1) * 100;
    }

    retArr.push(tick);
  }
  return retArr;
};

export const getBulkBacktestDetails = (
  bulkFetchBacktest: FetchBulkBacktests,
  {
    sinceYearFilter,
    selectedYearFilter,
    FILTER_NOT_SELECTED_VALUE,
    displayPairsArr,
  }: {
    sinceYearFilter: string;
    selectedYearFilter: string;
    FILTER_NOT_SELECTED_VALUE: string;
    displayPairsArr: DisplayPairsItem[];
  }
) => {
  if (!bulkFetchBacktest || !bulkFetchBacktest.data) return null;

  const balanceTicksArr: object[] = [];
  const yearsUsedInBacktest = new Set();

  let backtestKeyWithMostKlines = "";
  let mostKlinesCount = 0;

  bulkFetchBacktest.equity_curves.forEach((item) => {
    for (const [key, value] of Object.entries(item)) {
      if (value.length > mostKlinesCount) {
        mostKlinesCount = value.length;
        backtestKeyWithMostKlines = key;
      }
    }
  });

  bulkFetchBacktest.equity_curves.forEach((item) => {
    for (const [key, equityCurveTicks] of Object.entries(item)) {
      if (key === backtestKeyWithMostKlines) {
        equityCurveTicks.forEach((balance) => {
          const timestampToDate = new Date(balance[KLINE_OPEN_TIME_KEY]);
          yearsUsedInBacktest.add(timestampToDate.getFullYear());

          if (
            selectedYearFilter !== FILTER_NOT_SELECTED_VALUE &&
            String(timestampToDate.getFullYear()) !== selectedYearFilter
          ) {
            return;
          }

          if (
            sinceYearFilter !== FILTER_NOT_SELECTED_VALUE &&
            timestampToDate.getFullYear() < Number(sinceYearFilter)
          ) {
            return;
          }

          const portfolioValues = getMassSimFindTicks(
            bulkFetchBacktest,
            balance.kline_open_time
          );

          balanceTicksArr.push({
            kline_open_time: balance.kline_open_time,
            ...portfolioValues,
          });
        });
      }
    }
  });

  const returns = convertBalanceTicksToReturnTicks(balanceTicksArr);

  const eqCurveStatistics = getEquityCurveStatistics(
    bulkFetchBacktest,
    returns,
    displayPairsArr
  );

  const equityCurves = convertBalanceTicksToEqCurve(balanceTicksArr).map(
    (item) => {
      const combinedEqTick = binarySearch(
        eqCurveStatistics["multiStrategyReturnsCurve"],
        item[KLINE_OPEN_TIME_KEY],
        (item) => item[KLINE_OPEN_TIME_KEY]
      );

      const combinedEqTickValue =
        combinedEqTick === null ? null : combinedEqTick[EQUITY_KEY];

      return {
        ...item,
        [COMBINED_STRATEGY_DATA_KEY]: combinedEqTickValue,
      };
    }
  );
  return {
    equityCurves: equityCurves,
    years: Array.from(yearsUsedInBacktest),
    ...eqCurveStatistics,
    datasets: getDatasetsInBulkBacktest(bulkFetchBacktest),
  };
};

const getNumWinningAndLosingStrats = (totalReturnsByStrat: object) => {
  let numOfWinning = 0;
  let numOfLosing = 0;

  for (const [_, value] of Object.entries(totalReturnsByStrat)) {
    if (value < 1) {
      numOfLosing += 1;
    } else {
      numOfWinning += 1;
    }
  }

  return {
    numOfWinningStrats: numOfWinning,
    numOfLosingStrata: numOfLosing,
  };
};

export const getDatasetsInBulkBacktest = (
  bulkFetchBacktest: FetchBulkBacktests
) => {
  const ret = [] as string[];

  for (const [_, value] of Object.entries(
    bulkFetchBacktest.id_to_dataset_name_map
  )) {
    ret.push(value);
  }
  return ret;
};

export const getMultiStrategyTotalReturn = (totalReturnArr: object[]) => {
  if (totalReturnArr.length === 0) {
    return null;
  }
  const lastItem = totalReturnArr[totalReturnArr.length - 1];
  return lastItem[EQUITY_KEY];
};

export const getBestTotalReturn = (totalReturnsDict: object) => {
  let max = -1000;
  for (const [key, value] of Object.entries(totalReturnsDict)) {
    if (key === KLINE_OPEN_TIME_KEY) {
      continue;
    }

    if (value > max) {
      max = value;
    }
  }
  return (max - 1) * 100;
};

export const getWorstTotalReturn = (totalReturnsDict: object) => {
  let min = 1000;
  for (const [key, value] of Object.entries(totalReturnsDict)) {
    if (key === KLINE_OPEN_TIME_KEY) {
      continue;
    }

    if (value < min) {
      min = value;
    }
  }
  return (min - 1) * 100;
};

export const getMedianTotalReturn = (totalReturnsDict: object) => {
  const totalReturns: number[] = [];

  for (const [key, value] of Object.entries(totalReturnsDict)) {
    if (key === KLINE_OPEN_TIME_KEY) {
      continue;
    }

    totalReturns.push(value);
  }
  if (totalReturns.length === 0) return null;
  totalReturns.sort((a, b) => a - b);
  return (totalReturns[Math.floor(totalReturns.length / 2)] - 1) * 100;
};

export const findTradesByYear = (trades: Trade[], targetYear: number) => {
  const ret: Trade[] = [];

  trades.forEach((item) => {
    const year = new Date(item.open_time).getUTCFullYear();

    if (targetYear === year) {
      ret.push(item);
    }
  });

  return ret;
};

export const getBacktestFormDefaults = () => {
  return {
    backtestName: "",
    useTimeBasedClose: false,
    useProfitBasedClose: false,
    useStopLossBasedClose: false,
    klinesUntilClose: 0,
    tradingFees: 0.1,
    slippage: 0.001,
    shortFeeHourly: 0.00165888 / 100,
    takeProfitThresholdPerc: 0,
    stopLossThresholdPerc: 0,
    backtestDataRange: [70, 100],
  };
};

export const getBacktestFormDefaultKeys = () => {
  return {
    backtestName: "backtestName",
    useTimeBasedClose: "useTimeBasedClose",
    useStopLossBasedClose: "useStopLossBasedClose",
    klinesUntilClose: "klinesUntilClose",
    slippage: "slippage",
    useProfitBasedClose: "useProfitBasedClose",
    shortFeeHourly: "shortFeeHourly",
    takeProfitThresholdPerc: "takeProfitThresholdPerc",
    stopLossThresholdPerc: "stopLossThresholdPerc",
    backtestDataRange: "backtestDataRange",
    tradingFees: "tradingFees",
  };
};

export const BACKTEST_FORM_LABELS = {
  name: "Name (optional)",
  long_condition: "Long condition",
  long_short_buy_condition: "Buy condition",
  long_short_sell_condition: "Sell condition",
  close_condition: "Close condition",
  is_short_selling_strategy: "Is short selling strategy",
  use_time_based_close: "Use time based closing strategy",
  use_profit_based_close: "Use profit based close",
  klines_until_close: "Klines until close",
  take_profit_threshold: "Take profit threshold (%)",
  use_stop_loss_based_close: "Use stop loss based close",
  stop_loss_threshold: "Stop loss threshold (%)",
  trading_fees: "Trading fees (%)",
  slippage: "Slippage (%)",
  shorting_fees_hourly: "Shorting fees (%) hourly",
  backtest_data_range: "Backtest data range (%)",
};
