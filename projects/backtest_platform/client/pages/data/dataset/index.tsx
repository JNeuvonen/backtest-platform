import React, { useEffect } from "react";
import { ChakraTabs } from "../../../components/layout/Tabs";
import { DatasetInfoPage } from "./info";
import { DatasetEditorPage } from "./editor";
import useQueryParams from "../../../hooks/useQueryParams";
import { useAppContext } from "../../../context/app";
import { LAYOUT } from "../../../utils/constants";

const TAB_LABELS = ["Info", "Editor"];
const TABS = [<DatasetInfoPage key={"1"} />, <DatasetEditorPage key={"2"} />];

interface TabQueryParams {
  defaultTab: string | undefined;
}

export const DatasetIndex = () => {
  const { defaultTab } = useQueryParams<TabQueryParams>();
  const { setInnerSideNavWidth } = useAppContext();

  useEffect(() => {
    setInnerSideNavWidth(LAYOUT.inner_side_nav_width_px);
    return () => setInnerSideNavWidth(0);
  }, []);
  return (
    <div>
      <ChakraTabs
        labels={TAB_LABELS}
        tabs={TABS}
        defaultTab={defaultTab ? Number(defaultTab) : 0}
      />
    </div>
  );
};
