import React from "react";
import { ChakraTabs } from "../../components/layout/Tabs";
import { DatasetInfoPage } from "./info";
import { DatasetEditorPage } from "./editor";
import useQueryParams from "../../hooks/useQueryParams";
import { DatasetModelPage } from "./model";

const TAB_LABELS = ["Info", "Editor", "Create model"];
const TABS = [<DatasetInfoPage />, <DatasetEditorPage />, <DatasetModelPage />];

interface TabQueryParams {
  defaultTab: string | undefined;
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
