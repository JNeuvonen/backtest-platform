import { off } from "process";
import { FetchBulkBacktests } from "../clients/queries/response-types";
import { binarySearch } from "./algo";
import { getKeysCount } from "./object";

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
  balances: object[],
  returns: object[],
  backtestKeyWithMostKlines: string
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

  const totalStrats = getKeysCount(returns[0]) - 1;

  const totalReturnsByStrat = {};

  for (const [key] of Object.entries(endBalances)) {
    totalReturnsByStrat[key] = 1;
  }

  for (let i = 0; i < returns.length; ++i) {
    const returnsTick = returns[i];

    const tick = {};
    const roundReturns = {};

    for (const [key, value] of Object.entries(returnsTick)) {
      if (key === KLINE_OPEN_TIME_KEY) {
        tick[KLINE_OPEN_TIME_KEY] = value;
        continue;
      }
      const coeff = value === 0 ? 1 : value;

      roundReturns[key] = (multiStratCurrBalance / totalStrats) * coeff;
      totalReturnsByStrat[key] = totalReturnsByStrat[key] * coeff;
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
  }: {
    sinceYearFilter: string;
    selectedYearFilter: string;
    FILTER_NOT_SELECTED_VALUE: string;
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
    balanceTicksArr,
    returns,
    backtestKeyWithMostKlines
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
