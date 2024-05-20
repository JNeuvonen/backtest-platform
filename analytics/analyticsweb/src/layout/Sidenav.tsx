import { PATHS } from "../utils";
import { MdDashboardCustomize } from "react-icons/md";
import { FaChartLine } from "react-icons/fa";
import { Link } from "react-router-dom";
import { useActivePath } from "src/hooks";

const SIDE_NAV_ITEMS = [
  {
    link: "Live",
    icon: MdDashboardCustomize,
    path: PATHS.dashboard,
  },
  {
    link: "Strategies",
    icon: FaChartLine,
    path: PATHS.strategies,
  },
];

export const SideNav = () => {
  const { activePath } = useActivePath({ tabPathDepth: 1 });
  return (
    <nav className="side-nav">
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
    </nav>
  );
};
