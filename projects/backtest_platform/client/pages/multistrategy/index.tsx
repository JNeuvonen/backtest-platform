import {
  Heading,
  MenuButton,
  MenuItem,
  Spinner,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import React from "react";
import { BacktestDatagrid } from "../../components/data-grid/Backtest";
import { ChakraMenu } from "../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa";
import {
  CreateMultiStrategyBacktest,
  MultiStrategyBacktest,
} from "../../components/CreateMultiStrategyBacktest";
import {
  useMultiStrategyBacktests,
  useRuleBasedMassBacktests,
} from "../../clients/queries/queries";
import { createMultiStrategyBacktest } from "../../clients/requests";
import {
  MultiStrategyBacktestBody,
  RuleBasedMassBacktestBody,
} from "../../clients/queries/response-types";

export const MultiStrategyBacktestPage = () => {
  const newBacktestDrawer = useDisclosure();
  const onDeleteMode = useDisclosure();
  const toast = useToast();

  const backtestsQuery = useRuleBasedMassBacktests();
  const multiStratBacktests = useMultiStrategyBacktests();

  const onSubmit = async (values: MultiStrategyBacktest) => {
    if (!backtestsQuery.data) return;

    const backtests: RuleBasedMassBacktestBody[] = [];

    backtestsQuery.data.forEach((item) => {
      if (values.selectedStrategyIds.includes(item.id)) {
        backtests.push(JSON.parse(item.body_json_dump));
      }
    });

    const body: MultiStrategyBacktestBody = {
      name: values.backtestName,
      strategies: backtests,
    };

    const res = await createMultiStrategyBacktest(body);
    if (res.status === 200) {
      toast({
        title: "Multistrategy backtest started",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      newBacktestDrawer.onClose();
    }
  };

  return (
    <div>
      <CreateMultiStrategyBacktest
        drawerControls={newBacktestDrawer}
        onSubmit={onSubmit}
      />
      <div>
        <Heading size={"lg"}>Multistrategy backtest</Heading>
      </div>

      <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
        <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
          <MenuItem icon={<FaFileImport />} onClick={newBacktestDrawer.onOpen}>
            New
          </MenuItem>
        </ChakraMenu>
      </div>

      <div style={{ marginTop: "8px" }}>
        <BacktestDatagrid
          backtests={multiStratBacktests.data || []}
          onDeleteMode={onDeleteMode}
        />
      </div>
    </div>
  );
};
