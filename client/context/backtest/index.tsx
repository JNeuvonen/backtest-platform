import { UseDisclosureReturn, useDisclosure } from "@chakra-ui/react";
import React, { useContext } from "react";
import { ReactNode, createContext } from "react";
import { BacktestUXManager } from "./modals";

interface BacktestContextType {
  createNewDrawer: UseDisclosureReturn;
}

interface BacktestProvidersProps {
  children: ReactNode;
}

export const BacktestContext = createContext<BacktestContextType>(
  {} as BacktestContextType
);

export const BacktestProvider: React.FC<BacktestProvidersProps> = ({
  children,
}) => {
  const createNewDrawer = useDisclosure();
  return (
    <BacktestContext.Provider
      value={{
        createNewDrawer,
      }}
    >
      <BacktestUXManager />
      {children}
    </BacktestContext.Provider>
  );
};

export const useBacktestContext = () => useContext(BacktestContext);
