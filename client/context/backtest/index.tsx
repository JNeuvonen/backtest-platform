import { UseDisclosureReturn, useDisclosure } from "@chakra-ui/react";
import React, { useContext } from "react";
import { ReactNode, createContext } from "react";
import { usePathParams } from "../../hooks/usePathParams";
import {
  useDatasetQuery,
  useDatasetsBacktests,
} from "../../clients/queries/queries";
import { UseQueryResult } from "@tanstack/react-query";
import { BacktestObject, Dataset } from "../../clients/queries/response-types";
import { useForceUpdate } from "../../hooks/useForceUpdate";
import { BacktestForm } from "./backtest-form";
import { ShowColumnsModal } from "./columns-modal";
import { RunPythonModal } from "./run-python";

interface BacktestContextType {
  createNewDrawer: UseDisclosureReturn;
  showColumnsModal: UseDisclosureReturn;
  priceColumnPopover: UseDisclosureReturn;
  targetColumnPopover: UseDisclosureReturn;
  klineOpenTimePopover: UseDisclosureReturn;
  runPythonModal: UseDisclosureReturn;
  datasetBacktestsQuery: UseQueryResult<BacktestObject[] | null, unknown>;
  datasetQuery: UseQueryResult<Dataset | null, unknown>;
  forceUpdate: () => void;
  datasetName: string;
}

interface BacktestProvidersProps {
  children: ReactNode;
}

type PathParams = {
  datasetName: string;
};

export const BacktestContext = createContext<BacktestContextType>(
  {} as BacktestContextType
);

export const BacktestProvider: React.FC<BacktestProvidersProps> = ({
  children,
}) => {
  const { datasetName } = usePathParams<PathParams>();
  const datasetQuery = useDatasetQuery(datasetName);
  const datasetBacktestsQuery = useDatasetsBacktests(
    datasetQuery.data?.id || undefined
  );
  const createNewDrawer = useDisclosure();
  const showColumnsModal = useDisclosure();
  const targetColumnPopover = useDisclosure();
  const priceColumnPopover = useDisclosure();
  const klineOpenTimePopover = useDisclosure();
  const runPythonModal = useDisclosure();

  const forceUpdate = useForceUpdate();

  return (
    <BacktestContext.Provider
      value={{
        createNewDrawer,
        datasetBacktestsQuery,
        datasetQuery,
        forceUpdate,
        showColumnsModal,
        targetColumnPopover,
        priceColumnPopover,
        datasetName,
        klineOpenTimePopover,
        runPythonModal,
      }}
    >
      <BacktestForm />
      <ShowColumnsModal />
      <RunPythonModal />
      {children}
    </BacktestContext.Provider>
  );
};

export const useBacktestContext = () => useContext(BacktestContext);
