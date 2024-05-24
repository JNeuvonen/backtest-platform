export const getDiffToPresentFormatted = (date: Date) => {
  if (!(date instanceof Date)) {
    throw new Error("The provided value is not a valid Date object.");
  }

  const now = new Date();
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
