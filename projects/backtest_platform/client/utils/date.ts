export const getDateStr = (ts: number | string) => {
  return new Date(ts).toLocaleDateString();
};

export function formatSecondsIntoTime(seconds: number): string {
  const weeks = Math.floor(seconds / 604800);
  const days = Math.floor((seconds % 604800) / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  let result = "";

  if (weeks > 0) {
    result += `${weeks}w `;
  }
  if (days > 0 || (weeks > 0 && hours > 0)) {
    result += `${days}d `;
  }
  if (hours > 0 || weeks > 0 || days > 0) {
    result += `${hours}h`;
  }
  return result.trim();
}

export function convertMillisToDateDict(kline_open_time: number) {
  const date = new Date(kline_open_time);
  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate(),
  };
}
