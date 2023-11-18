import React from "react";
import { InnerSideNav } from "../components/layout/InnerSideNav";
import { SideNavItem } from "../Components/Layout/SideNav";
import { PATHS } from "../utils/constants";
import { Outlet } from "react-router-dom";

const SIDE_NAV_ITEMS: SideNavItem[] = [
  { link: "Available", path: PATHS.datasets.subpaths.available.index },
  { link: "Binance", path: PATHS.datasets.subpaths.binance.index },
];

export const DatasetsPage = () => {
  return (
    <div className="layout__container-inner-side-nav">
      <InnerSideNav
        sideNavItems={SIDE_NAV_ITEMS}
        pathActiveItemDepth={2}
        fallbackPath={PATHS.datasets.subpaths.available.index}
      />
      <Outlet />
    </div>
  );
};
