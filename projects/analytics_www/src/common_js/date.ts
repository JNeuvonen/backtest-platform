export const getDiffToPresentFormatted = (date: Date) => {
  if (!(date instanceof Date)) {
    throw new Error("The provided value is not a valid Date object.");
  }

  const now = getUTCDate();
  const diffInMilliseconds = now.getTime() - date.getTime();

  const millisecondsInSecond = 1000;
  const millisecondsInMinute = millisecondsInSecond * 60;
  const millisecondsInHour = millisecondsInMinute * 60;
  const millisecondsInDay = millisecondsInHour * 24;
  const millisecondsInWeek = millisecondsInDay * 7;

  if (diffInMilliseconds < millisecondsInMinute) {
    const seconds = Math.floor(diffInMilliseconds / millisecondsInSecond);
    return `${seconds} ${seconds === 1 ? "second" : "seconds"}`;
  } else if (diffInMilliseconds < millisecondsInHour) {
    const minutes = Math.floor(diffInMilliseconds / millisecondsInMinute);
    return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
  } else if (diffInMilliseconds < millisecondsInDay) {
    const hours = Math.floor(diffInMilliseconds / millisecondsInHour);
    return `${hours} ${hours === 1 ? "hour" : "hours"}`;
  } else if (diffInMilliseconds < millisecondsInWeek) {
    const days = Math.floor(diffInMilliseconds / millisecondsInDay);
    return `${days} ${days === 1 ? "day" : "days"}`;
  } else {
    const weeks = Math.floor(diffInMilliseconds / millisecondsInWeek);
    return `${weeks} ${weeks === 1 ? "week" : "weeks"}`;
  }
};

export function getUTCDate(): Date {
  const now = new Date();
  return new Date(now.getTime() + now.getTimezoneOffset() * 60000);
}

export const isSameMonth = (date1: Date, date2: Date) => {
  if (date1.getFullYear() !== date2.getFullYear()) {
    return false;
  }
  return date1.getMonth() === date2.getMonth();
};

export const isSameYear = (date1: Date, date2: Date) => {
  return date1.getFullYear() === date2.getFullYear();
};

export const isSame24h = (date1: Date, date2: Date) => {
  const diff = Math.abs(date1.getTime() - date2.getTime());
  return diff < 24 * 60 * 60 * 1000;
};

export const getDiffToPresentInHours = (date: Date) => {
  if (!(date instanceof Date)) {
    throw new Error("The provided value is not a valid Date object.");
  }

  const now = new Date();
  const diffInMilliseconds = now.getTime() - date.getTime();

  const millisecondsInSecond = 1000;
  const millisecondsInMinute = millisecondsInSecond * 60;
  const millisecondsInHour = millisecondsInMinute * 60;

  if (diffInMilliseconds < millisecondsInMinute) {
    const seconds = Math.floor(diffInMilliseconds / millisecondsInSecond);
    return `${seconds} ${seconds === 1 ? "second" : "seconds"}`;
  } else if (diffInMilliseconds < millisecondsInHour) {
    const minutes = Math.floor(diffInMilliseconds / millisecondsInMinute);
    return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
  } else {
    const hours = Math.floor(diffInMilliseconds / millisecondsInHour);
    return `${hours} ${hours === 1 ? "hour" : "hours"}`;
  }
};
