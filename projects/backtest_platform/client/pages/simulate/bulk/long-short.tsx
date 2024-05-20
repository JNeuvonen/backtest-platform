import { Heading, MenuButton, MenuItem, useDisclosure } from "@chakra-ui/react";
import React from "react";
import { ChakraMenu } from "../../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa6";
import { BacktestDatagrid } from "../../../components/data-grid/Backtest";
import { useMassBacktestContext } from "../../../context/masspairtrade";

export const BulkLongShortSimPage = () => {
  const massPairTradeContext = useMassBacktestContext();
  return (
    <div>
      <div>
        <Heading size={"lg"}>Mass pair-trade</Heading>
      </div>

      <div style={{ display: "flex", gap: "16px" }}>
        <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
          <MenuItem
            icon={<FaFileImport />}
            onClick={() => massPairTradeContext.createNewDrawer.onOpen()}
          >
            New
          </MenuItem>
        </ChakraMenu>
      </div>
      <div style={{ marginTop: "8px" }}>
        <BacktestDatagrid
          backtests={massPairTradeContext.longShortBacktestsQuery.data || []}
          onDeleteMode={massPairTradeContext.onDeleteMode}
        />
      </div>
    </div>
  );
};
