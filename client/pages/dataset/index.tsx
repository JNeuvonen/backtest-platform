import React from "react";
import { PATHS, PATH_KEYS } from "../../utils/constants";
import { SideNavItem } from "../../components/layout/SideNav";
import { InnerSideNav } from "../../components/layout/InnerSideNav";
import { Outlet } from "react-router-dom";
import { usePathParams } from "../../hooks/usePathParams";

const SIDE_NAV_ITEMS: SideNavItem[] = [
  { link: "Info", path: PATHS.datasets.info },
  { link: "Editor", path: PATHS.datasets.editor },
];

export const DatasetIndex = () => {
  const item = usePathParams({ key: PATH_KEYS.dataset }) as string;
  return (
    <div className="layout__container-inner-side-nav">
      <InnerSideNav
        sideNavItems={SIDE_NAV_ITEMS}
        pathActiveItemDepth={3}
        fallbackPath={PATHS.datasets.info}
        formatPath={(path) => {
          return path.replace(PATH_KEYS.dataset, item);
        }}
      />
      <Outlet />
    </div>
  );
};
