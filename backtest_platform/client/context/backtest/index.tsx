import { UseDisclosureReturn, useDisclosure } from "@chakra-ui/react";
import React, { useContext, useState } from "react";
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
import { FilterBacktestDrawer } from "./filter-backtests";
import { ConfirmDeleteSelectedModal } from "./confirm-delete";
import { DeployStrategyForm } from "../../pages/simulate/dataset/backtest/DeployStrategyForm";

interface BacktestContextType {
  createNewDrawer: UseDisclosureReturn;
  filterDrawer: UseDisclosureReturn;
  showColumnsModal: UseDisclosureReturn;
  priceColumnPopover: UseDisclosureReturn;
  targetColumnPopover: UseDisclosureReturn;
  klineOpenTimePopover: UseDisclosureReturn;
  confirmDeleteSelectedModal: UseDisclosureReturn;
  runPythonModal: UseDisclosureReturn;
  onDeleteMode: UseDisclosureReturn;
  datasetBacktestsQuery: UseQueryResult<BacktestObject[] | null, unknown>;
  datasetQuery: UseQueryResult<Dataset | null, unknown>;
  forceUpdate: () => void;
  datasetName: string;
  selectedBacktests: number[];
  selectBacktest: (backtestId: number) => void;
  resetSelection: () => void;
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
  const filterDrawer = useDisclosure();
  const onDeleteMode = useDisclosure();
  const confirmDeleteSelectedModal = useDisclosure();

  const forceUpdate = useForceUpdate();
  const [selectedBacktests, setSelectedBacktests] = useState<number[]>([]);

  const selectBacktest = (backtestId: number) => {
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
        filterDrawer,
        onDeleteMode,
        selectedBacktests,
        selectBacktest,
        resetSelection,
        confirmDeleteSelectedModal,
      }}
    >
      <BacktestForm />
      <ShowColumnsModal />
      <RunPythonModal />
      <FilterBacktestDrawer />
      <ConfirmDeleteSelectedModal />
      {children}
    </BacktestContext.Provider>
  );
};

export const useBacktestContext = () => useContext(BacktestContext);
