import React from "react";
import { InnerSideNav } from "../components/layout/InnerSideNav";
import { SideNavItem } from "../components/Layout/SideNav";
import { PATHS } from "../utils/constants";
import { Outlet } from "react-router-dom";

const SIDE_NAV_ITEMS: SideNavItem[] = [
  { link: "All", path: PATHS.datasets.subpaths.available.path },
  { link: "Stock Market", path: PATHS.datasets.subpaths.stock_market.path },
  { link: "Binance", path: PATHS.datasets.subpaths.binance.path },
];

export const DatasetsPage = () => {
  return (
    <div className="layout__container-inner-side-nav">
      <InnerSideNav
        sideNavItems={SIDE_NAV_ITEMS}
        pathActiveItemDepth={2}
        fallbackPath={PATHS.datasets.subpaths.stock_market.path}
      />
      <Outlet />
    </div>
  );
};
