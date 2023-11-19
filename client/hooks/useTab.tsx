import { useState } from "react";

export function getNthPathSegment(n: number) {
  const pathArray = window.location.pathname.split("/");
  return pathArray[n];
}
export function replaceNthPathSegment(n: number, newSegment: string) {
  const pathArray = window.location.pathname.split("/");
  pathArray[n] = newSegment;
  return pathArray.join("/");
}

interface Props {
  pathActiveItemDepth: number;
}

export const useTab = ({ pathActiveItemDepth }: Props) => {
  const [activePathItem, setActivePathItem] = useState(
    getNthPathSegment(pathActiveItemDepth),
  );

  const setNewTab = (tab: string) => {
    setActivePathItem(tab);
  };
  return {
    activePathItem,
    setNewTab,
  };
};
