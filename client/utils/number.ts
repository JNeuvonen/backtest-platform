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
