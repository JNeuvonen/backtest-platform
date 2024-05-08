import { Heading, MenuButton, MenuItem } from "@chakra-ui/react";
import React from "react";
import { ChakraMenu } from "../../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa";
import { BacktestDatagrid } from "../../../components/data-grid/Backtest";
import { useMLBasedBacktestContext } from "../../../context/mlbasedbacktest";
import { usePathParams } from "../../../hooks/usePathParams";
import ExternalLink from "../../../components/ExternalLink";
import { getDatasetInfoPagePath } from "../../../utils/navigate";
import {
  useDatasetQuery,
  useDatasetsBacktests,
} from "../../../clients/queries/queries";
import { DOM_EVENT_CHANNELS } from "../../../utils/constants";
import { useMessageListener } from "../../../hooks/useMessageListener";

interface PathParams {
  datasetName: string;
}
export const MachineLearningBacktestPage = () => {
  const mlBasedBacktestContext = useMLBasedBacktestContext();
  const { datasetName } = usePathParams<PathParams>();
  const datasetQuery = useDatasetQuery(datasetName);
  const datasetBacktestsQuery = useDatasetsBacktests(
    datasetQuery.data?.id || undefined
  );

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_component,
    messageCallback: () => {
      datasetBacktestsQuery.refetch();
    },
  });

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
          backtests={datasetBacktestsQuery.data || []}
          onDeleteMode={mlBasedBacktestContext.onDeleteMode}
        />
      </div>
    </div>
  );
};
