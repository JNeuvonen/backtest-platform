export const roundNumberFloorStr = (
  num: number | null,
  decimalPlaces: number,
  format: boolean = false,
): string => {
  if (num === null) {
    return "";
  }
  const factor = Math.pow(10, decimalPlaces);
  const rounded = Math.floor(num * factor) / factor;

  if (rounded === 0 && decimalPlaces < 8) {
    return roundNumberFloorStr(num, decimalPlaces + 1, format);
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

export const roundNumberFloor = (
  num: number,
  decimalPlaces: number,
): number => {
  const factor = Math.pow(10, decimalPlaces);
  const rounded = Math.floor(num * factor) / factor;

  if (rounded === 0 && decimalPlaces < 8) {
    return roundNumberFloor(num, decimalPlaces + 1);
  }

  return rounded;
};
