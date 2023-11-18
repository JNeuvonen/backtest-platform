import React, { useEffect } from "react";
import { SideNavItem } from "./SideNav";
import { Link, useLocation } from "react-router-dom";
import { getNthPathSegment } from "../../hooks/useTab";

interface Props {
  sideNavItems: SideNavItem[];
  pathActiveItemDepth: number;
  fallbackPath: string;
}

export const InnerSideNav = ({
  sideNavItems,
  pathActiveItemDepth,
  fallbackPath,
}: Props) => {
  const location = useLocation();
  const [activePath, setActivePath] = React.useState("");

  useEffect(() => {
    let currentPath = getNthPathSegment(pathActiveItemDepth);
    currentPath = !currentPath ? fallbackPath : currentPath;
    setActivePath(currentPath);
  }, [location, pathActiveItemDepth, fallbackPath]);

  console.log(window.location.pathname);
  return (
    <div className="layout__page-side-nav">
      {sideNavItems.map((item) => {
        const styleClassName = "layout__container-inner-side-nav__item";
        const activeClassName = "layout__container-inner-side-nav__item-active";
        return (
          <Link to={item.path} key={item.link}>
            <div
              className={
                activePath === item.path
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
