import React from "react";
import { ChakraTabs } from "../../../components/layout/Tabs";
import { DatasetInfoPage } from "./info";
import { DatasetEditorPage } from "./editor";
import useQueryParams from "../../../hooks/useQueryParams";
import { DatasetModelCreatePage } from "../model/create";

const TAB_LABELS = ["Info", "Editor", "Models"];
const TABS = [
  <DatasetInfoPage key={"1"} />,
  <DatasetEditorPage key={"2"} />,
  <DatasetModelCreatePage key={"3"} />,
];

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
