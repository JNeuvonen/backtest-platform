import React from "react";
import {
  Badge,
  Heading,
  MenuButton,
  MenuItem,
  Spinner,
} from "@chakra-ui/react";
import { ChakraMenu } from "../../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa6";
import { useBacktestContext } from "../../../context/backtest";
import { BacktestDatagrid } from "../../../components/data-grid/Backtest";

export const SimulateDatasetIndex = () => {
  const { createNewDrawer, datasetQuery, datasetBacktestsQuery } =
    useBacktestContext();

  if (!datasetQuery.data) {
    return <Spinner />;
  }

  return (
    <div>
      <Heading size={"lg"}>Backtest</Heading>

      <div style={{ marginTop: "16px" }}>
        Price column:{" "}
        <Badge colorScheme="green">{datasetQuery.data.price_col}</Badge>
      </div>
      <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
        <MenuItem icon={<FaFileImport />} onClick={createNewDrawer.onOpen}>
          New
        </MenuItem>
        <MenuItem icon={<FaFileImport />}>Update target column</MenuItem>
      </ChakraMenu>

      <div>
        {datasetBacktestsQuery.data ? (
          <BacktestDatagrid backtests={datasetBacktestsQuery.data} />
        ) : (
          <Spinner />
        )}
      </div>
    </div>
  );
};
