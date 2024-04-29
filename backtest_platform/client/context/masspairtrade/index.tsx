import { UseDisclosureReturn, useDisclosure } from "@chakra-ui/react";
import React, { ReactNode, createContext, useContext, useState } from "react";

interface MassPairTradeProviderProps {
  children: ReactNode;
}

interface MassPairTradeContextType {
  createNewDrawer: UseDisclosureReturn;
  onDeleteMode: UseDisclosureReturn;
  selectLongShortBacktest: (backtestId: number) => void;
  resetSelection: () => void;
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
      }}
    >
      {children}
    </MassPairTradeContext.Provider>
  );
};

export const useMassBacktestContext = () => useContext(MassPairTradeContext);
