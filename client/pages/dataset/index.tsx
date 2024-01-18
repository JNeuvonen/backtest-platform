import React from "react";
import { PATHS, PATH_KEYS } from "../../utils/constants";
import { SideNavItem } from "../../components/layout/SideNav";
import { InnerSideNav } from "../../components/layout/InnerSideNav";
import { Outlet } from "react-router-dom";
import { usePathParams } from "../../hooks/usePathParams";
import { PiComputerTowerFill } from "react-icons/pi";

const SIDE_NAV_ITEMS: SideNavItem[] = [
  {
    link: "Info",
    path: PATHS.datasets.info,
    icon: <PiComputerTowerFill size={20} />,
  },
  {
    link: "Editor",
    path: PATHS.datasets.editor,
    icon: <PiComputerTowerFill size={20} />,
  },
];

interface RouteParams {
  datasetName: string;
}

export const DatasetIndex = () => {
  const { datasetName } = usePathParams<RouteParams>();
  return (
    <div className="layout__container-inner-side-nav">
      <InnerSideNav
        sideNavItems={SIDE_NAV_ITEMS}
        pathActiveItemDepth={3}
        fallbackPath={PATHS.datasets.info}
        formatPath={(path) => {
          return path.replace(PATH_KEYS.dataset, datasetName);
        }}
      />
      <Outlet />
    </div>
  );
};
