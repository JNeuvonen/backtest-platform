import React from "react";
import { BrowseDatasetsPage } from "./Datasets";
import { InnerSideNav } from "../components/layout/InnerSideNav";
import { BsDatabase } from "react-icons/bs";
import { PATHS, PATH_KEYS } from "../utils/constants";
import usePath from "../hooks/usePath";
import { usePathParams } from "../hooks/usePathParams";
import { OutletContent } from "../components/layout/TwoNavsIndent";

const SIDE_NAV_ITEMS = [
  {
    link: "Datasets",
    icon: <BsDatabase />,
    path: PATHS.data.dataset.index,
  },
  {
    link: "Models",
    icon: <BsDatabase />,
    path: PATHS.data.model.index,
  },
];

export const DataRouteIndex = () => {
  const { path } = usePath();
  const { datasetName } = usePathParams<{ datasetName: string }>();
  return (
    <div>
      {path !== PATHS.data.index && (
        <InnerSideNav
          sideNavItems={SIDE_NAV_ITEMS}
          pathActiveItemDepth={2}
          formatPath={(item) => {
            const path = item.replace(PATH_KEYS.dataset, datasetName);
            return path;
          }}
        />
      )}
      {path === PATHS.data.index && (
        <div>
          <BrowseDatasetsPage />
        </div>
      )}
      {path !== PATHS.data.index && <OutletContent />}
    </div>
  );
};
