import { UseDisclosureReturn, useDisclosure } from "@chakra-ui/react";
import React, { ReactNode, createContext, useContext } from "react";

interface MLBasedBacktestProviderProps {
  children: ReactNode;
}

interface MLBasedBacktestContextType {
  createNewDrawer: UseDisclosureReturn;
  onDeleteMode: UseDisclosureReturn;
}

export const MLBasedBacktestContext = createContext<MLBasedBacktestContextType>(
  {} as MLBasedBacktestContextType
);

export const MLBasedBacktestProvider: React.FC<
  MLBasedBacktestProviderProps
> = ({ children }: MLBasedBacktestProviderProps) => {
  const createNewDrawer = useDisclosure();
  const onDeleteMode = useDisclosure();
  return (
    <MLBasedBacktestContext.Provider
      value={{
        createNewDrawer,
        onDeleteMode,
      }}
    >
      {children}
    </MLBasedBacktestContext.Provider>
  );
};

export const useMLBasedBacktestContext = () =>
  useContext(MLBasedBacktestContext);
