export const replaceNthPathItem = (nthItemFromEnd: number, newItem: string) => {
  const pathParts = window.location.pathname.split("/");
  pathParts[pathParts.length - 1 - nthItemFromEnd] = newItem;
  return pathParts.join("/");
};
