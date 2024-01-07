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
