/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */

export function isObjectEmpty<T extends object>(obj: T): boolean {
  return Object.keys(obj).length === 0;
}

export function areAllValuesNull<T extends object>(obj: T): boolean {
  return Object.values(obj).every((value) => value === null);
}

export function areAllNestedValuesNull<T extends Record<string, any>>(
  obj: T
): boolean {
  return Object.values(obj).every(
    (nestedObj) =>
      typeof nestedObj === "object" &&
      nestedObj !== null &&
      areAllValuesNull(nestedObj)
  );
}

export function isOneNestedValueTrue<T extends Record<string, any>>(
  obj: T
): boolean {
  return Object.values(obj).some(
    (nestedObj) =>
      typeof nestedObj === "object" &&
      nestedObj !== null &&
      Object.values(nestedObj).some((value) => value === true)
  );
}

export function getKeysCount<T extends object>(obj: T): number {
  if (!obj) return 0;
  return Object.keys(obj).length;
}

export function getNonnullEntriesCount<T extends object>(obj: T): number {
  let count = 0;
  for (const [_key, value] of Object.entries(obj)) {
    if (value !== null) {
      count += 1;
    }
    if (typeof value === "object" && value !== null) {
      count += getNonnullEntriesCount(value);
      count -= 1;
    }
  }
  return count;
}
