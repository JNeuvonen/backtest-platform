export const makeUniDimensionalTableRows = (
  rows: string[] | number[] | (string | number)[]
) => {
  const ret: (string | number)[][] = [[]];
  rows.forEach((item) => {
    ret[0].push(item);
  });
  return ret;
};
