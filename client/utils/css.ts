export const createScssClassName = (nestedSelectors: string[]) => {
  return nestedSelectors.join("__");
};
