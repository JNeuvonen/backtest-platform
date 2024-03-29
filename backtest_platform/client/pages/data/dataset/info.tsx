import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDatasetQuery } from "../../../clients/queries/queries";
import {
  Box,
  IconButton,
  MenuButton,
  MenuItem,
  Spinner,
  useToast,
} from "@chakra-ui/react";
import { ChakraModal } from "../../../components/chakra/modal";
import { useModal } from "../../../hooks/useOpen";
import { BUTTON_VARIANTS } from "../../../theme";
import { ConfirmInput } from "../../../components/form/ConfirmInput";
import { buildRequest } from "../../../clients/fetch";
import { URLS } from "../../../clients/endpoints";
import { replaceNthPathItem } from "../../../utils/navigate";
import { useMessageListener } from "../../../hooks/useMessageListener";
import {
  CODE,
  DOM_EVENT_CHANNELS,
  DOM_IDS,
  NullFillStrategy,
} from "../../../utils/constants";
import { ColumnModal } from "../../../components/RenameColumnModal";
import { RunPythonOnAllCols } from "../../../components/RunPythonOnAllCols";
import { FormSubmitBar } from "../../../components/form/FormSubmitBar";
import { createPythonCode } from "../../../utils/str";
import { usePathParams } from "../../../hooks/usePathParams";
import { ConfirmModal } from "../../../components/form/Confirm";
import {
  execPythonOnDataset,
  saveDatasetFile,
  setKlineOpenTimeColumn,
  setTargetColumnReq,
  updatePriceColumnReq,
} from "../../../clients/requests";
import { getValueById } from "../../../utils/dom";
import { ChakraPopover } from "../../../components/chakra/popover";
import { CreateCopyPopover } from "../../../components/CreateCopyPopover";
import { SelectColumnPopover } from "../../../components/SelectTargetColumnPopover";
import { getDatasetColumnOptions } from "../../../utils/dataset";
import { ChakraMenu } from "../../../components/chakra/Menu";
import {
  AddIcon,
  DownloadIcon,
  EditIcon,
  ExternalLinkIcon,
  HamburgerIcon,
  RepeatIcon,
} from "@chakra-ui/icons";
import { DatasetDataGrid } from "../../../components/data-grid/Dataset";
import { CellClickedEvent, CellValueChangedEvent } from "ag-grid-community";

type DatasetDetailParams = {
  datasetName: string;
};

const { GET_DATASET_EXAMPLE, DATASET_SYMBOL, INDENT } = CODE;

const getCodeDefaultValue = () => {
  return createPythonCode([
    `${GET_DATASET_EXAMPLE}`,
    "",
    "#This example transforms all columns into a simple moving average of 30 last datapoints",
    `#for item in ${DATASET_SYMBOL}.columns:`,
    `#${INDENT}dataset[item] = dataset[item].rolling(window=30).mean()`,
  ]);
};

export const DatasetInfoPage = () => {
  const { datasetName } = usePathParams<DatasetDetailParams>();
  const [inputDatasetName, setInputDatasetName] = useState(datasetName);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [code, setCode] = useState<string>(getCodeDefaultValue());

  const toast = useToast();
  const navigate = useNavigate();

  const columnModal = useModal();
  const runPythonModal = useModal();
  const confirmRunPythonModal = useModal();
  const targetColumnPopover = useModal();
  const klineOpenTimePopover = useModal();
  const createCopyPopover = useModal();
  const backtestPriceColumnPopover = useModal();

  const { data, isLoading, refetch } = useDatasetQuery(datasetName);
  const [editedRows, setEditedRows] = useState<object[]>([]);

  const columnOnClickFunction = (selectedColumn: string) => {
    setSelectedColumn(selectedColumn);
    columnModal.setIsOpen(true);
  };

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_dataset,
    messageCallback: refetch,
  });

  if (isLoading) {
    return <Spinner />;
  }

  const dataset = data;

  if (!dataset) {
    return <Box>Page is not available</Box>;
  }
  const renameDataset = (newDatasetName: string) => {
    const res = buildRequest({
      url: URLS.set_dataset_name(datasetName),
      method: "PUT",
      payload: { new_dataset_name: newDatasetName },
    });

    res
      .then((res) => {
        if (res?.status === 200) {
          toast({
            title: "Changed the datasets name",
            status: "success",
            duration: 5000,
            isClosable: true,
          });
          setInputDatasetName(newDatasetName);
          navigate(replaceNthPathItem(1, newDatasetName));
        } else {
          toast({
            title: "Failed to change the datasets name",
            status: "error",
            duration: 5000,
            isClosable: true,
          });
        }
      })
      .catch((error) => {
        toast({
          title: "Error",
          description: error?.message,
          status: "error",
          duration: 5000,
          isClosable: true,
        });
      });
  };

  const submitExecPythonOnDataset = async () => {
    const nullFillStrategy = getValueById(
      DOM_IDS.select_null_fill_strat
    ) as NullFillStrategy;

    const res = await execPythonOnDataset(datasetName, code, nullFillStrategy);
    if (res.status === 200) {
      toast({
        title: "Executed python code",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      runPythonModal.modalClose();
      confirmRunPythonModal.modalClose();
      refetch();
    }
  };

  const setTargetColumn = async (targetCol: string) => {
    const res = await setTargetColumnReq(datasetName, targetCol);
    if (res.status === 200) {
      toast({
        title: "Changed target column",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      targetColumnPopover.onClose();
      refetch();
    }
  };

  const setBacktestPriceColumn = async (value: string) => {
    const res = await updatePriceColumnReq(datasetName, value);
    if (res.status === 200) {
      toast({
        title: "Changed price column",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      backtestPriceColumnPopover.onClose();
      refetch();
    }
  };

  const columns = dataset.columns;

  return (
    <div>
      <ConfirmModal
        {...confirmRunPythonModal}
        onClose={confirmRunPythonModal.modalClose}
        onConfirm={submitExecPythonOnDataset}
      />
      <ChakraModal
        isOpen={columnModal.isOpen}
        title={selectedColumn}
        onClose={columnModal.modalClose}
        modalContentStyle={{
          minWidth: "max-content",
          minHeight: "80%",
          maxWidth: "70%",
          marginTop: "10vh",
        }}
      >
        <ColumnModal
          columnName={selectedColumn}
          setColumnName={setSelectedColumn}
          datasetName={datasetName}
          close={columnModal.modalClose}
        />
      </ChakraModal>
      <ChakraModal
        isOpen={runPythonModal.isOpen}
        title={"Run python"}
        onClose={runPythonModal.modalClose}
        modalContentStyle={{
          minWidth: "max-content",
          minHeight: "50%",
          maxWidth: "90%",
          marginTop: "10vh",
        }}
        footerContent={
          <FormSubmitBar
            cancelCallback={runPythonModal.modalClose}
            submitCallback={confirmRunPythonModal.modalOpen}
          />
        }
      >
        <RunPythonOnAllCols code={code} setCode={setCode} />
      </ChakraModal>
      <Box
        display={"flex"}
        justifyContent={"space-between"}
        alignItems={"center"}
      >
        <Box>
          <ConfirmInput
            inputCurrent={inputDatasetName}
            setInputCurrent={setInputDatasetName}
            defaultValue={datasetName}
            newInputCallback={renameDataset}
            disallowedCharsRegex={/\s/}
            message={
              <>
                <span>
                  Are you sure you want to rename dataset to{" "}
                  <b>{inputDatasetName}</b>?
                </span>
              </>
            }
          />
        </Box>
        <Box display={"flex"} gap={"16px"}>
          <div>
            <ChakraMenu
              menuButton={
                <MenuButton
                  as={IconButton}
                  aria-label="Options"
                  icon={<HamburgerIcon />}
                  variant={BUTTON_VARIANTS.grey}
                />
              }
            >
              <MenuItem icon={<AddIcon />} onClick={createCopyPopover.onOpen}>
                Create copy
              </MenuItem>
              <MenuItem
                icon={<ExternalLinkIcon />}
                onClick={backtestPriceColumnPopover.onOpen}
              >
                Set backtest price column
              </MenuItem>
              <MenuItem
                icon={<RepeatIcon />}
                onClick={klineOpenTimePopover.onOpen}
              >
                Set candle open time column
              </MenuItem>
              <MenuItem
                icon={<EditIcon />}
                onClick={targetColumnPopover.onOpen}
              >
                Set target column
              </MenuItem>
              <MenuItem icon={<EditIcon />} onClick={runPythonModal.modalOpen}>
                Modify the dataset with python
              </MenuItem>

              <MenuItem
                icon={<DownloadIcon />}
                onClick={() => {
                  saveDatasetFile(datasetName);
                }}
              >
                Download the dataset
              </MenuItem>
            </ChakraMenu>
            <ChakraPopover
              useArrow={false}
              isOpen={createCopyPopover.isOpen}
              setOpen={createCopyPopover.onOpen}
              onClose={createCopyPopover.onClose}
              headerText="Create a copy of the dataset"
              body={
                <CreateCopyPopover
                  datasetName={datasetName}
                  successCallback={createCopyPopover.onClose}
                  cancelCallback={createCopyPopover.onClose}
                />
              }
            />

            <ChakraPopover
              isOpen={backtestPriceColumnPopover.isOpen}
              setOpen={backtestPriceColumnPopover.onOpen}
              onClose={backtestPriceColumnPopover.onClose}
              headerText="Set backtest price column"
              useArrow={false}
              body={
                <SelectColumnPopover
                  options={getDatasetColumnOptions(dataset)}
                  placeholder={dataset.price_col}
                  selectCallback={setBacktestPriceColumn}
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
                  options={getDatasetColumnOptions(dataset)}
                  placeholder={dataset.timeseries_col}
                  selectCallback={(newCol) => {
                    setKlineOpenTimeColumn(newCol, datasetName, () => {
                      toast({
                        title: "Changed candle open time column",
                        status: "info",
                        duration: 5000,
                        isClosable: true,
                      });
                      refetch();
                      klineOpenTimePopover.onClose();
                    });
                  }}
                />
              }
            />
            <ChakraPopover
              isOpen={targetColumnPopover.isOpen}
              setOpen={targetColumnPopover.onOpen}
              onClose={targetColumnPopover.onClose}
              headerText="Set target column"
              useArrow={false}
              body={
                <SelectColumnPopover
                  options={getDatasetColumnOptions(dataset)}
                  placeholder={dataset.target_col}
                  selectCallback={setTargetColumn}
                />
              }
            />
          </div>
        </Box>
      </Box>
      <Box
        marginTop={"16px"}
        style={{
          marginLeft: "-32px",
          width: "calc(100% + 32px)",
        }}
      >
        <DatasetDataGrid
          columnDefs={columns.map((item) => {
            return {
              headerName: item,
              field: item,
              sortable: false,
              editable: item !== dataset.timeseries_col ? true : false,
            };
          })}
          onCellClicked={(e: CellClickedEvent) => {
            columnOnClickFunction(e.column.getColId());
          }}
          handleCellValueChanged={(rowData: CellValueChangedEvent) => {
            setEditedRows([...editedRows, rowData.data]);
          }}
          maxRows={dataset.row_count}
          columnLabels={columns}
          datasetName={datasetName}
        />
      </Box>
    </div>
  );
};
