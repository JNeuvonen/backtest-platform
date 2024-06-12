import { Heading, MenuButton, MenuItem, useDisclosure } from "@chakra-ui/react";
import React, { useState } from "react";
import { ChakraMenu } from "../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa";
import { GiSelect } from "react-icons/gi";
import { SelectUniverseModal } from "../../components/SelectUniverseModal";
import { OptionType } from "../../components/SelectFilter";
import { MultiValue } from "react-select";
import { CreateMassRuleBasedSim } from "../../components/CreateMassRuleBasedSim";

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
      <CreateMassRuleBasedSim drawerControls={newBacktestDrawer} />
      <div>
        <Heading size={"lg"}>Rule-based on universe</Heading>
      </div>
      <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
        <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
          <MenuItem icon={<FaFileImport />} onClick={newBacktestDrawer.onOpen}>
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
    </div>
  );
};
