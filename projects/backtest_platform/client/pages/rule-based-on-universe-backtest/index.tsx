import {
  Heading,
  MenuButton,
  MenuItem,
  Tooltip,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import React, { useState } from "react";
import { ChakraMenu } from "../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa";
import { GiSelect } from "react-icons/gi";
import { SelectUniverseModal } from "../../components/SelectUniverseModal";
import { OptionType } from "../../components/SelectFilter";
import { MultiValue } from "react-select";
import {
  BacktestOnUniverseFormValues,
  CreateMassRuleBasedSim,
} from "../../components/CreateMassRuleBasedSim";
import { createRuleBasedMassBacktest } from "../../clients/requests";
import { useRuleBasedMassBacktests } from "../../clients/queries/queries";
import { BacktestDatagrid } from "../../components/data-grid/Backtest";

export const RuleBasedSimOnUniverseBacktest = () => {
  const selectUniverseModal = useDisclosure();
  const newBacktestDrawer = useDisclosure();
  const [stockMarketSymbols, setStockMarketSymbols] = useState<
    MultiValue<OptionType>
  >([]);
  const [cryptoSymbols, setCryptoSymbols] = useState<MultiValue<OptionType>>(
    []
  );
  const [candleInterval, setCandleInterval] = useState("1d");
  const backtestsQuery = useRuleBasedMassBacktests();
  const onDeleteMode = useDisclosure();
  const toast = useToast();

  const onCreateBacktest = async (values: BacktestOnUniverseFormValues) => {
    const backtestBody = {
      name: values.backtestName,
      candle_interval: candleInterval,
      datasets: cryptoSymbols.map((item) => item.value),
      start_date: values.startDate?.toISOString() || null,
      end_date: values.endDate?.toISOString() || null,
      data_transformations: values.dataTransformations,
      klines_until_close: values.klinesUntilClose,
      open_trade_cond: values.enterCriteria,
      close_trade_cond: values.exitCriteria,
      fetch_latest_data: values.useLatestData,
      is_cryptocurrency_datasets: true,
      is_short_selling_strategy: values.isShortSellingStrategy,
      use_time_based_close: values.useTimeBasedClose,
      use_profit_based_close: values.useProfitBasedClose,
      use_stop_loss_based_close: values.useStopLossBasedClose,
      stop_loss_threshold_perc: values.stopLossThresholdPerc,
      short_fee_hourly: values.shortFeeHourly,
      trading_fees_perc: values.tradingFees,
      slippage_perc: values.slippage,
      allocation_per_symbol: values.allocationPerSymbol,
      take_profit_threshold_perc: values.takeProfitThresholdPerc,
    };

    const res = await createRuleBasedMassBacktest(backtestBody);

    if (res.status === 200) {
      toast({
        title: "Started backtest",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      newBacktestDrawer.onClose();
    }
  };

  return (
    <div>
      <SelectUniverseModal
        modalControls={selectUniverseModal}
        onSelectCryptoSymbols={(items: MultiValue<OptionType>) => {
          setCryptoSymbols(items);
        }}
        onSelectStockMarketSymbols={(items: MultiValue<OptionType>) => {
          setStockMarketSymbols(items);
        }}
        onSelectCandleInterval={(newCandleInterval: string) =>
          setCandleInterval(newCandleInterval)
        }
      />
      <CreateMassRuleBasedSim
        drawerControls={newBacktestDrawer}
        onSubmit={onCreateBacktest}
      />
      <div>
        <Heading size={"lg"}>Rule-based on universe</Heading>
      </div>
      <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
        <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
          <MenuItem
            icon={<FaFileImport />}
            onClick={newBacktestDrawer.onOpen}
            isDisabled={cryptoSymbols.length === 0}
          >
            New
          </MenuItem>
          <MenuItem icon={<GiSelect />} onClick={selectUniverseModal.onOpen}>
            Select universe
          </MenuItem>
        </ChakraMenu>
        <div>
          Universe size: {stockMarketSymbols.length + cryptoSymbols.length}
        </div>
      </div>

      <div style={{ marginTop: "8px" }}>
        <BacktestDatagrid
          backtests={backtestsQuery.data || []}
          onDeleteMode={onDeleteMode}
        />
      </div>
    </div>
  );
};
