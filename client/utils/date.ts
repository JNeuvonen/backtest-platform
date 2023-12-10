export const getDateStr = (ts: number | string) => {
  return new Date(ts).toLocaleDateString();
};
