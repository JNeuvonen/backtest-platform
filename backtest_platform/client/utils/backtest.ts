import { FetchBulkBacktests } from "../clients/queries/response-types";
import { binarySearch } from "./algo";
import { getKeysCount } from "./object";

const KLINE_OPEN_TIME_KEY = "kline_open_time";

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
      if (key === "kline_open_time") {
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

  let multiStratCurrBalance = 10000;
  const multiStratBalanceTicks: object[] = [];

  const totalStrats = getKeysCount(returns[0]) - 1;

  for (let i = 0; i < returns.length; ++i) {
    const returnsTick = returns[i];

    const tick = {};
    const roundReturns = {};

    for (const [key, value] of Object.entries(returnsTick)) {
      if (key === "kline_open_time") {
        tick["kline_open_time"] = value;
        continue;
      }
      const coeff = value === 0 ? 1 : value;

      roundReturns[key] = (multiStratCurrBalance / totalStrats) * coeff;
    }

    let tickBalance = 0;

    for (const [_, value] of Object.entries(roundReturns)) {
      tickBalance += value as number;
    }

    multiStratCurrBalance = tickBalance;
    tick["equity"] = multiStratCurrBalance;
    multiStratBalanceTicks.push(tick);
  }

  return {
    endBalances: endBalances,
    multiStrategyEquityCurve: multiStratBalanceTicks,
    multiStrategyReturnsCurve: convertBalanceTicksToEqCurve(
      multiStratBalanceTicks
    ),
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
      if (key === "kline_open_time") {
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
      if (key === "kline_open_time") {
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
        if (key === "kline_open_time") {
          continue;
        }
        helper[key] = 1;
      }
      continue;
    }

    const tick = {
      kline_open_time: currItem["kline_open_time"],
    };

    for (const [key, value] of Object.entries(currItem)) {
      if (key === "kline_open_time") {
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

export const getMassSimEquityCurvesData = (
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

  return {
    equityCurves: convertBalanceTicksToEqCurve(balanceTicksArr),
    years: Array.from(yearsUsedInBacktest),
    ...eqCurveStatistics,
  };
};
