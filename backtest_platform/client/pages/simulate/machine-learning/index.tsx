import { Heading, MenuButton, MenuItem } from "@chakra-ui/react";
import React from "react";
import { ChakraMenu } from "../../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa";
import { BacktestDatagrid } from "../../../components/data-grid/Backtest";
import { useMLBasedBacktestContext } from "../../../context/mlbasedbacktest";
import { usePathParams } from "../../../hooks/usePathParams";
import ExternalLink from "../../../components/ExternalLink";
import { getDatasetInfoPagePath } from "../../../utils/navigate";

interface PathParams {
  datasetName: string;
}
export const MachineLearningBacktestPage = () => {
  const mlBasedBacktestContext = useMLBasedBacktestContext();
  const { datasetName } = usePathParams<PathParams>();
  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div>
          <Heading size={"lg"}>ML strategy - {datasetName}</Heading>
        </div>

        <ExternalLink
          to={getDatasetInfoPagePath(datasetName)}
          linkText={"Dataset"}
        />
      </div>

      <div style={{ display: "flex", gap: "16px" }}>
        <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
          <MenuItem
            icon={<FaFileImport />}
            onClick={() => mlBasedBacktestContext.createNewDrawer.onOpen()}
          >
            New
          </MenuItem>
        </ChakraMenu>
      </div>
      <div style={{ marginTop: "8px" }}>
        <BacktestDatagrid
          backtests={[]}
          onDeleteMode={mlBasedBacktestContext.onDeleteMode}
        />
      </div>
    </div>
  );
};
