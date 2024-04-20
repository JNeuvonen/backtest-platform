export const roundNumberDropRemaining = (
  num: number,
  decimalPlaces: number,
  format: boolean = false
): string => {
  const factor = Math.pow(10, decimalPlaces);
  const rounded = Math.floor(num * factor) / factor;

  if (rounded === 0 && decimalPlaces < 8) {
    return roundNumberDropRemaining(num, decimalPlaces + 1, format);
  }

  if (format) {
    const absRounded = Math.abs(rounded);
    const suffix = rounded >= 0 ? "" : "-";

    if (absRounded >= 1e9) {
      return `${suffix}${(absRounded / 1e9).toFixed(2)}B`;
    } else if (absRounded >= 1e6) {
      return `${suffix}${(absRounded / 1e6).toFixed(2)}M`;
    } else if (absRounded >= 1e3) {
      return `${suffix}${(absRounded / 1e3).toFixed(2)}K`;
    }
  }

  return rounded.toString();
};

export const getNumberArrayMean = (numbers: number[]) => {
  if (numbers.length === 0) return NaN;
  let sum = 0;
  for (let i = 0; i < numbers.length; ++i) {
    sum += numbers[i];
  }
  return sum / numbers.length;
};

export const getArrayMedian = (numbers: number[]) => {
  if (numbers.length === 0) return NaN;
  const midPoint = Math.floor(numbers.length / 2);
  return numbers[midPoint];
};

export const getArrayMin = (numbers: number[]) => {
  if (numbers.length === 0) return NaN;
  let min = 999999999999;

  for (let i = 0; i < numbers.length; ++i) {
    if (numbers[i] < min) {
      min = numbers[i];
    }
  }
  return min;
};

export const getArrayMax = (numbers: number[]) => {
  if (numbers.length === 0) return NaN;
  let max = -999999999999;

  for (let i = 0; i < numbers.length; ++i) {
    if (numbers[i] > max) {
      max = numbers[i];
    }
  }
  return max;
};

export const calculateStdDevAndMean = (numbers: number[]) => {
  if (numbers.length === 0) {
    return {
      stdDev: NaN,
      mean: NaN,
      variance: NaN,
    };
  }

  const mean = getNumberArrayMean(numbers);
  let varianceSum = 0;
  for (let i = 0; i < numbers.length; ++i) {
    const diffToMean = mean - numbers[i];
    varianceSum += diffToMean * diffToMean;
  }
  const variance = varianceSum / numbers.length;
  return {
    stdDev: Math.sqrt(variance),
    mean,
    variance,
  };
};

export const calculateZScore = (x: number, mean: number, stdDev: number) => {
  return (x - mean) / stdDev;
};

export interface NormalDistributionItems {
  zScoreRangeStart: number;
  count: number;
  label: string;
}
export const getNormalDistributionItems = (
  numbers: number[],
  zStart = -3,
  bars = 100
) => {
  const ret = [] as NormalDistributionItems[];

  const { mean, stdDev } = calculateStdDevAndMean(numbers);
  const copy = [...numbers];
  copy.sort((a, b) => a - b);
  let jStart = 0;

  const zEnd = Math.abs(zStart);
  const increment = (zEnd * 2) / bars;

  for (let i = zStart; i < zEnd; i += increment) {
    let itemsInZCategory = 0;
    let categorySum = 0;

    for (let j = jStart; j < copy.length; ++j) {
      const zScore = calculateZScore(copy[j], mean, stdDev);
      if (zScore > i && zScore <= i + increment) {
        jStart = j;
        itemsInZCategory += 1;
        categorySum += copy[j];
      }

      if (zScore > i + 0.06) {
        jStart = j;

        break;
      }
    }

    ret.push({
      zScoreRangeStart: i,
      count: itemsInZCategory,
      label:
        itemsInZCategory !== 0
          ? String(roundNumberDropRemaining(categorySum / itemsInZCategory, 2))
          : "",
    });
  }
  return ret;
};

export const removeDuplicates = (numbers: number[]) => {
  return [...new Set(numbers)];
};

export type NumericKey<T> = {
  [K in keyof T]: T[K] extends number ? K : never;
}[keyof T];

export const calculateMean = <
  T extends Record<K, number>,
  K extends NumericKey<T>,
>(
  arr: T[],
  key: K
): number => {
  if (arr.length === 0) return 0;
  return arr.reduce((acc, curr) => acc + curr[key], 0) / arr.length;
};
