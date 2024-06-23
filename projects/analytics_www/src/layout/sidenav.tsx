import { MOBILE_WIDTH_CUTOFF, PATHS, TOP_BAR_HEIGHT } from "../utils";
import { MdDashboardCustomize } from "react-icons/md";
import { FaChartLine, FaUserCircle } from "react-icons/fa";
import { Link, useNavigate } from "react-router-dom";
import { useActivePath, useWinDimensions } from "src/hooks";
import { useEffect, useState } from "react";
import { IconButton } from "@chakra-ui/react";
import { Blur } from "src/components";
import { RxHamburgerMenu } from "react-icons/rx";
import { useAuth0 } from "@auth0/auth0-react";
import {
  COLOR_BG_PRIMARY,
  COLOR_BG_SECONDARY_SHADE_ONE,
  COLOR_BG_TERTIARY,
} from "src/theme";
import { BsCashStack } from "react-icons/bs";

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
  {
    link: "Assets",
    icon: BsCashStack,
    path: PATHS.assets,
  },
];

const ProfileButton = () => {
  const { user } = useAuth0();
  const navigate = useNavigate();
  const { activePath } = useActivePath({ tabPathDepth: 1 });

  return (
    <button
      className="profile-button"
      onClick={() => navigate(PATHS.profile)}
      style={{
        backgroundColor:
          `/${activePath}` === PATHS.profile
            ? COLOR_BG_SECONDARY_SHADE_ONE
            : "",
      }}
    >
      <div>Settings</div>
      <div>
        {user?.picture ? (
          <img
            src={user?.picture}
            alt={user?.name}
            className="profile-picture"
          />
        ) : (
          <FaUserCircle className="fallback-icon" width={32} height={32} />
        )}
      </div>
    </button>
  );
};

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

  const isPathActive = (path: string) => {
    if (activePath === "/" && path === "/") {
      return true;
    }
    if (activePath === "/" && path !== "/") {
      return false;
    }
    return path.includes(activePath);
  };

  return (
    <>
      <Blur
        isEnabled={!hideSidenav && width < MOBILE_WIDTH_CUTOFF}
        onClickCallback={() => {
          setHideSidenav(true);
        }}
      />

      {hideSidenav && (
        <div
          style={{
            position: "sticky",
            top: "0px",
            width: "100%",
            background: COLOR_BG_PRIMARY,
            height: TOP_BAR_HEIGHT,
            display: "flex",
            alignItems: "center",
            paddingLeft: "8px",
            zIndex: 1000,
          }}
        >
          <IconButton
            icon={<RxHamburgerMenu />}
            aria-label={"Menu button"}
            height={"30px"}
            width={"30px"}
            background={COLOR_BG_TERTIARY}
            onClick={() => setHideSidenav(false)}
          />
        </div>
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
                      isPathActive(item.path)
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
          <div>
            <ProfileButton />
          </div>
        </nav>
      )}
    </>
  );
};
