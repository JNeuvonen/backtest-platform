import React, { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { getNthPathSegment } from ".";

export const useActivePath = ({ tabPathDepth }: { tabPathDepth: number }) => {
  const location = useLocation();
  const [activePath, setActivePath] = React.useState("");

  useEffect(() => {
    let currentPath = getNthPathSegment(tabPathDepth);
    currentPath = !currentPath ? "/" : currentPath;
    setActivePath(currentPath);
  }, [location, tabPathDepth]);

  return { activePath };
};
