import React from "react";
import { ChakraTabs } from "../../components/layout/Tabs";
import { DatasetInfoPage } from "./info";
import { DatasetEditorPage } from "./editor";
import useQueryParams from "../../hooks/useQueryParams";

const TAB_LABELS = ["Info", "Editor", "Model"];
const TABS = [<DatasetInfoPage />, <DatasetEditorPage />, <div>model</div>];

interface TabQueryParams {
  defaultTab: number | undefined;
}

export const DatasetIndex = () => {
  const { defaultTab } = useQueryParams<TabQueryParams>();
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
