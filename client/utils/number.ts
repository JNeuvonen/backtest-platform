export const roundNumberDropRemaining = (
  num: number,
  decimalPlaces: number
): number => {
  const factor = Math.pow(10, decimalPlaces);
  const rounded = Math.floor(num * factor) / factor;

  if (rounded === 0 && decimalPlaces < 8) {
    return roundNumberDropRemaining(num, decimalPlaces + 1);
  }

  return rounded;
};

export const getNumberArrayMean = (numbers: number[]) => {
  if (numbers.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < numbers.length; ++i) {
    sum += numbers[i];
  }
  return sum / numbers.length;
};

export const getArrayMin = (numbers: number[]) => {
  let min = 999999999999;

  for (let i = 0; i < numbers.length; ++i) {
    if (numbers[i] < min) {
      min = numbers[i];
    }
  }
  return min;
};

export const getArrayMax = (numbers: number[]) => {
  let max = -999999999999;

  for (let i = 0; i < numbers.length; ++i) {
    if (numbers[i] > max) {
      max = numbers[i];
    }
  }
  return max;
};

export const calculateStdDevAndMean = (numbers: number[]) => {
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
export const getNormalDistributionItems = (numbers: number[]) => {
  const ret = [] as NormalDistributionItems[];

  const { mean, stdDev } = calculateStdDevAndMean(numbers);
  const copy = [...numbers];
  copy.sort((a, b) => a - b);
  let jStart = 0;
  for (let i = -3; i < 3; i += 0.06) {
    let itemsInZCategory = 0;
    let categorySum = 0;

    for (let j = jStart; j < copy.length; ++j) {
      const zScore = calculateZScore(copy[j], mean, stdDev);
      if (zScore > i && zScore <= i + 0.06) {
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
