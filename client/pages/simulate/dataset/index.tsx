import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  Badge,
  Heading,
  MenuButton,
  MenuItem,
  Spinner,
} from "@chakra-ui/react";
import { useDatasetQuery } from "../../../clients/queries/queries";
import { ChakraMenu } from "../../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa6";
import { useBacktestContext } from "../../../context/backtest";
import { BacktestDatagrid } from "../../../components/data-grid/Backtest";

type PathParams = {
  datasetName: string;
};

export const SimulateDatasetIndex = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { data } = useDatasetQuery(datasetName);
  const { createNewDrawer } = useBacktestContext();

  if (!data) {
    return <Spinner />;
  }

  return (
    <div>
      <Heading size={"lg"}>Backtest</Heading>

      <div style={{ marginTop: "16px" }}>
        Price column: <Badge colorScheme="green">{data.price_col}</Badge>
      </div>
      <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
        <MenuItem icon={<FaFileImport />} onClick={createNewDrawer.onOpen}>
          New
        </MenuItem>
        <MenuItem icon={<FaFileImport />}>Update target column</MenuItem>
      </ChakraMenu>

      <div>
        <BacktestDatagrid />
      </div>
    </div>
  );
};
