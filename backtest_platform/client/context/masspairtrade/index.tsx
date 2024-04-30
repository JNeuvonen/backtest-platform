import { UseDisclosureReturn, useDisclosure } from "@chakra-ui/react";
import React, { ReactNode, createContext, useContext, useState } from "react";
import { BulkLongShortCreateNew } from "./CreateNewDrawer";
import { useLongShortBacktests } from "../../clients/queries/queries";
import { UseQueryResult } from "@tanstack/react-query";
import { BacktestObject } from "../../clients/queries/response-types";

interface MassPairTradeProviderProps {
  children: ReactNode;
}

interface MassPairTradeContextType {
  createNewDrawer: UseDisclosureReturn;
  onDeleteMode: UseDisclosureReturn;
  selectLongShortBacktest: (backtestId: number) => void;
  resetSelection: () => void;
  longShortBacktestsQuery: UseQueryResult<BacktestObject[] | null, unknown>;
}

export const MassPairTradeContext = createContext<MassPairTradeContextType>(
  {} as MassPairTradeContextType
);

export const MassPairTradeProvider: React.FC<MassPairTradeProviderProps> = ({
  children,
}: MassPairTradeProviderProps) => {
  const createNewDrawer = useDisclosure();
  const [selectedBacktests, setSelectedBacktests] = useState<number[]>([]);
  const onDeleteMode = useDisclosure();
  const longShortBacktestsQuery = useLongShortBacktests();

  const selectLongShortBacktest = (backtestId: number) => {
    setSelectedBacktests((prevSelectedBacktests) => {
      if (prevSelectedBacktests.includes(backtestId)) {
        return prevSelectedBacktests.filter((item) => item !== backtestId);
      } else {
        return [...prevSelectedBacktests, backtestId];
      }
    });
  };

  const resetSelection = () => {
    setSelectedBacktests([]);
  };

  return (
    <MassPairTradeContext.Provider
      value={{
        createNewDrawer,
        selectLongShortBacktest,
        onDeleteMode,
        resetSelection,
        longShortBacktestsQuery,
      }}
    >
      <BulkLongShortCreateNew />
      {children}
    </MassPairTradeContext.Provider>
  );
};

export const useMassBacktestContext = () => useContext(MassPairTradeContext);
