export const getNumberDisplayColor = (num: number, zeroColor: string) => {
  if (num > 0) {
    return "green";
  }

  if (num < 0) {
    return "red";
  }

  return zeroColor;
};
