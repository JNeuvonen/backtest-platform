import { MOBILE_WIDTH_CUTOFF, PATHS } from "../utils";
import { MdDashboardCustomize } from "react-icons/md";
import { FaChartLine } from "react-icons/fa";
import { Link } from "react-router-dom";
import { useActivePath, useWinDimensions } from "src/hooks";
import { useEffect, useState } from "react";
import { Button, IconButton } from "@chakra-ui/react";
import { Blur } from "src/components";
import { RxHamburgerMenu } from "react-icons/rx";

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
  const { width } = useWinDimensions();
  const [hideSidenav, setHideSidenav] = useState(width < MOBILE_WIDTH_CUTOFF);

  useEffect(() => {
    if (width < MOBILE_WIDTH_CUTOFF) {
      setHideSidenav(true);
    } else {
      setHideSidenav(false);
    }
  }, [width]);

  return (
    <>
      <Blur
        isEnabled={!hideSidenav && width < MOBILE_WIDTH_CUTOFF}
        onClickCallback={() => {
          setHideSidenav(true);
        }}
      />

      {hideSidenav && (
        <IconButton
          icon={<RxHamburgerMenu />}
          style={{ position: "fixed" }}
          onClick={() => setHideSidenav(false)}
          aria-label={"Menu button"}
          marginLeft={"8px"}
          marginTop={"8px"}
        />
      )}

      {!hideSidenav && (
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
      )}
    </>
  );
};
