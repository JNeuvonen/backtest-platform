import { Button } from "@chakra-ui/react";
import React from "react";
import { InnerSideNav } from "../components/layout/InnerSideNav";
import { SideNavItem } from "../Components/Layout/SideNav";
import { PATHS } from "../utils/constants";

const SIDE_NAV_ITEMS: SideNavItem[] = [
  { link: "Available", path: PATHS.datasets.subpaths.available.index },
  { link: "Binance", path: PATHS.datasets.subpaths.binance.index },
];

export const DatasetsPage = () => {
  return (
    <div className="layout__container-inner-side-nav">
      <h1>Datasets</h1>
      <div>
        <Button>+ New Dataset</Button>
      </div>
      <h1>Available Datasets</h1>
      <InnerSideNav
        sideNavItems={SIDE_NAV_ITEMS}
        pathActiveItemDepth={2}
        fallbackPath={PATHS.datasets.subpaths.available.index}
      />
    </div>
  );
};
