import React from "react";
import {
  Badge,
  Heading,
  MenuButton,
  MenuItem,
  Spinner,
  useToast,
} from "@chakra-ui/react";
import { ChakraMenu } from "../../../components/chakra/Menu";
import { FaFileImport } from "react-icons/fa6";
import { useBacktestContext } from "../../../context/backtest";
import { BacktestDatagrid } from "../../../components/data-grid/Backtest";
import { FiTarget } from "react-icons/fi";
import { MdOutlinePriceCheck } from "react-icons/md";
import { CiTimer } from "react-icons/ci";
import { DiPython } from "react-icons/di";
import { CiViewColumn } from "react-icons/ci";
import { FaFilter } from "react-icons/fa";
import { MdDeleteForever } from "react-icons/md";
import { ChakraPopover } from "../../../components/chakra/popover";
import { SelectColumnPopover } from "../../../components/SelectTargetColumnPopover";
import { getDatasetColumnOptions } from "../../../utils/dataset";
import { FaUndoAlt } from "react-icons/fa";
import { GiSelect } from "react-icons/gi";
import {
  setBacktestPriceColumn,
  setKlineOpenTimeColumn,
  setTargetColumn,
} from "../../../clients/requests";
import { getParenthesisSize } from "../../../utils/content";

export const SimulateDatasetIndex = () => {
  const {
    createNewDrawer,
    datasetQuery,
    datasetBacktestsQuery,
    showColumnsModal,
    targetColumnPopover,
    priceColumnPopover,
    datasetName,
    klineOpenTimePopover,
    runPythonModal,
    filterDrawer,
    onDeleteMode,
    selectedBacktests,
    resetSelection,
    confirmDeleteSelectedModal,
  } = useBacktestContext();

  const toast = useToast();

  if (!datasetQuery.data) {
    return <Spinner />;
  }

  return (
    <div>
      <Heading size={"lg"}>Backtest {datasetName}</Heading>

      <div style={{ marginTop: "16px", display: "flex", gap: "16px" }}>
        <div>
          Target column:{" "}
          <Badge colorScheme="green">{datasetQuery.data.target_col}</Badge>
        </div>
        <div>
          Price column:{" "}
          <Badge colorScheme="green">{datasetQuery.data.price_col}</Badge>
        </div>
        <div>
          Timeseries column:{" "}
          <Badge colorScheme="green">{datasetQuery.data.timeseries_col}</Badge>
        </div>
      </div>
      <div style={{ display: "flex", gap: "16px" }}>
        <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
          <MenuItem icon={<FaFileImport />} onClick={createNewDrawer.onOpen}>
            New backtest
          </MenuItem>
          <MenuItem icon={<CiViewColumn />} onClick={showColumnsModal.onOpen}>
            Show columns
          </MenuItem>
          <MenuItem icon={<FaFilter />} onClick={filterDrawer.onOpen}>
            Filter
          </MenuItem>
          <MenuItem icon={<GiSelect />} onClick={onDeleteMode.onOpen}>
            Select
          </MenuItem>
          <MenuItem
            icon={<FaUndoAlt />}
            isDisabled={selectedBacktests.length === 0}
            onClick={() => {
              onDeleteMode.onClose();
              resetSelection();
            }}
          >
            Undo select {getParenthesisSize(selectedBacktests.length)}
          </MenuItem>
          <MenuItem
            icon={<MdDeleteForever />}
            onClick={() => {
              confirmDeleteSelectedModal.onOpen();
            }}
            isDisabled={selectedBacktests.length === 0}
          >
            Delete selected {getParenthesisSize(selectedBacktests.length)}
          </MenuItem>
        </ChakraMenu>

        <ChakraMenu menuButton={<MenuButton>Edit</MenuButton>}>
          <MenuItem icon={<FiTarget />} onClick={targetColumnPopover.onOpen}>
            Set target column
          </MenuItem>
          <MenuItem
            icon={<MdOutlinePriceCheck />}
            onClick={priceColumnPopover.onOpen}
          >
            Set price column
          </MenuItem>
          <MenuItem icon={<CiTimer />} onClick={klineOpenTimePopover.onOpen}>
            Set kline open time column
          </MenuItem>
          <MenuItem icon={<DiPython />} onClick={runPythonModal.onOpen}>
            Run python on dataset
          </MenuItem>
        </ChakraMenu>

        <div style={{ marginLeft: "100px", marginTop: "32px" }}>
          <ChakraPopover
            isOpen={targetColumnPopover.isOpen}
            setOpen={targetColumnPopover.onOpen}
            onClose={targetColumnPopover.onClose}
            headerText="Set target column"
            useArrow={false}
            body={
              <SelectColumnPopover
                options={getDatasetColumnOptions(datasetQuery.data)}
                placeholder={datasetQuery.data.target_col}
                selectCallback={(newCol) => {
                  setTargetColumn(newCol, datasetName, () => {
                    toast({
                      title: "Changed target column",
                      status: "info",
                      duration: 5000,
                      isClosable: true,
                    });
                    targetColumnPopover.onClose();
                    datasetQuery.refetch();
                  });
                }}
              />
            }
          />
          <ChakraPopover
            isOpen={priceColumnPopover.isOpen}
            setOpen={priceColumnPopover.onOpen}
            onClose={priceColumnPopover.onClose}
            headerText="Set backtest price column"
            useArrow={false}
            body={
              <SelectColumnPopover
                options={getDatasetColumnOptions(datasetQuery.data)}
                placeholder={datasetQuery.data.price_col}
                selectCallback={(newCol: string) => {
                  setBacktestPriceColumn(newCol, datasetName, () => {
                    toast({
                      title: "Changed price column",
                      status: "info",
                      duration: 5000,
                      isClosable: true,
                    });
                    priceColumnPopover.onClose();
                    datasetQuery.refetch();
                  });
                }}
              />
            }
          />
          <ChakraPopover
            isOpen={klineOpenTimePopover.isOpen}
            setOpen={klineOpenTimePopover.onOpen}
            onClose={klineOpenTimePopover.onClose}
            headerText="Set candle open time column"
            useArrow={false}
            body={
              <SelectColumnPopover
                options={getDatasetColumnOptions(datasetQuery.data)}
                placeholder={datasetQuery.data.timeseries_col}
                selectCallback={(newCol) => {
                  setKlineOpenTimeColumn(newCol, datasetName, () => {
                    toast({
                      title: "Changed candle open time column",
                      status: "info",
                      duration: 5000,
                      isClosable: true,
                    });
                    datasetQuery.refetch();
                    klineOpenTimePopover.onClose();
                  });
                }}
              />
            }
          />
        </div>
      </div>
      <div>
        {datasetBacktestsQuery.data ? (
          <BacktestDatagrid backtests={datasetBacktestsQuery.data} />
        ) : (
          <Spinner />
        )}
      </div>
    </div>
  );
};
