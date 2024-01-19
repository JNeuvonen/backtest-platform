import React, { CSSProperties } from "react";
import { TabList, Tab, Tabs, TabPanels, TabPanel } from "@chakra-ui/react";

interface Props {
  labels: string[];
  tabs: React.ReactNode[];
  style?: CSSProperties;
  defaultTab?: number;
}

export const ChakraTabs = ({
  labels,
  tabs,
  style = {},
  defaultTab = 0,
}: Props) => {
  return (
    <Tabs style={style} defaultIndex={defaultTab}>
      <TabList style={{ width: "max-content" }}>
        {labels.map((label, index) => (
          <Tab key={index}>{label}</Tab>
        ))}
      </TabList>

      <TabPanels>
        {tabs.map((tabContent, index) => (
          <TabPanel key={index}>{tabContent}</TabPanel>
        ))}
      </TabPanels>
    </Tabs>
  );
};
