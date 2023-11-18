import React from "react";
import { BsDatabase } from "react-icons/bs";
import { useLocation } from "react-router-dom";
import { Link } from "react-router-dom";
import { LINKS, PATHS } from "../../utils/constants";

const SIDE_NAV_ITEMS = [
  { link: LINKS.datasets, icon: BsDatabase, path: PATHS.datasets },
  { link: LINKS.simulate, icon: BsDatabase, path: PATHS.simulate },
];

export const SideNav = () => {
  const location = useLocation();
  const currentPath = location.pathname;
  return (
    <div className="side-nav">
      <div className="side-nav__nav-items">
        {SIDE_NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <Link to={item.path}>
              {" "}
              <div
                className={
                  currentPath === item.path
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
