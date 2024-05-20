import { PATHS } from "../utils";
import { MdDashboardCustomize } from "react-icons/md";

const SIDE_NAV_ITEMS = [
  {
    link: "Dashboard",
    icon: MdDashboardCustomize,
    path: PATHS.dashboard,
  },
];

export const SideNav = () => {
  return <nav className="side-nav">nav</nav>;
};
