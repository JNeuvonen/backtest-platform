export function safeDivide(
  numerator: number,
  denominator: number,
  fallback: number,
): number {
  return denominator === 0 ? fallback : numerator / denominator;
}

export const getPercDifference = (num1: number, num2: number): number => {
  const difference = num2 - num1;
  const average = (num1 + num2) / 2;
  const percentageDifference = safeDivide(difference, average, 0) * 100;
  return percentageDifference;
};

export const getRateOfChangePerc = (num1: number, num2: number): number => {
  const roc = safeDivide(num1, num2, 1) - 1;
  return roc * 100;
};
