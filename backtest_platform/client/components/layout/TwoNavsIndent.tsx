import React from "react";
import { useAppContext } from "../../context/app";
import { Outlet } from "react-router-dom";
export const OutletContent = () => {
  const { innerSideNavWidth } = useAppContext();
  return (
    <div style={{ marginLeft: innerSideNavWidth }}>
      <Outlet />
    </div>
  );
};
