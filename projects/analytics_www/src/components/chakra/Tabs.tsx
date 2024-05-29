import React, { CSSProperties } from "react";
import { TabList, Tab, Tabs, TabPanels, TabPanel } from "@chakra-ui/react";
import { MARGIN_TOP_TABS } from "src/layout";

interface Props {
  labels: string[];
  tabs: React.ReactNode[];
  style?: CSSProperties;
  defaultTab?: number;
  top?: number;
}

export const ChakraTabs = ({
  labels,
  tabs,
  style = {},
  defaultTab = 0,
  top = 0,
}: Props) => {
  return (
    <div style={style}>
      <Tabs defaultIndex={defaultTab}>
        <div
          style={{
            position: "fixed",
            top: 0,
            paddingTop: top,
            paddingBottom: top,
            width: "100%",
            zIndex: 1000,
            background: "#212121",
          }}
        >
          <TabList style={{ width: "max-content" }}>
            {labels.map((label, index) => (
              <Tab key={index}>{label}</Tab>
            ))}
          </TabList>
        </div>

        <div style={{ marginTop: MARGIN_TOP_TABS }}>
          <TabPanels>
            {tabs.map((tabContent, index) => (
              <TabPanel key={index}>{tabContent}</TabPanel>
            ))}
          </TabPanels>
        </div>
      </Tabs>
    </div>
  );
};
