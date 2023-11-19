import React from "react";
import { BsDatabase } from "react-icons/bs";
import { Link } from "react-router-dom";
import { LINKS, PATHS } from "../../utils/constants";
import { useActivePath } from "../../hooks/useActivePath";

export interface SideNavItem {
  link: string;
  icon?: JSX.Element;
  path: string;
}

const SIDE_NAV_ITEMS = [
  {
    link: LINKS.datasets,
    icon: BsDatabase,
    path: PATHS.datasets.path + "/" + PATHS.datasets.subpaths.available.path,
  },
  { link: LINKS.simulate, icon: BsDatabase, path: PATHS.simulate.path },
];

export const SideNav = () => {
  const { activePath } = useActivePath({ tabPathDepth: 1 });
  return (
    <div className="side-nav">
      <div className="side-nav__nav-items">
        {SIDE_NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <Link to={item.path} key={item.path}>
              <div
                className={
                  item.path.includes(activePath)
                    ? "side-nav__item side-nav__item-active"
                    : "side-nav__item"
                }
              >
                {<Icon />}
                {item.link}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
};
