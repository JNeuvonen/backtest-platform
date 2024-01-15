import React from "react";
import { SideNavItem } from "./SideNav";
import { Link } from "react-router-dom";
import { useActivePath } from "../../hooks/useActivePath";

interface Props {
  sideNavItems: SideNavItem[];
  pathActiveItemDepth: number;
  fallbackPath: string;
  formatPath?: (path: string) => string;
}

export const InnerSideNav = ({
  sideNavItems,
  pathActiveItemDepth,
  formatPath,
}: Props) => {
  const { activePath } = useActivePath({ tabPathDepth: pathActiveItemDepth });
  return (
    <div className="layout__page-side-nav">
      {sideNavItems.map((item) => {
        const styleClassName = "layout__container-inner-side-nav__item";
        const activeClassName = "layout__container-inner-side-nav__item-active";
        return (
          <Link
            to={formatPath ? formatPath(item.path) : item.path}
            key={item.link}
          >
            <div
              className={
                item.path.includes(activePath)
                  ? styleClassName + " " + activeClassName
                  : styleClassName
              }
            >
              {item.link}
            </div>
          </Link>
        );
      })}
    </div>
  );
};
