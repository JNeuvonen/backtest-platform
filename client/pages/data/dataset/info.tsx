import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDatasetQuery } from "../../../clients/queries/queries";
import { Box, Button, Select, Spinner, useToast } from "@chakra-ui/react";
import { GenericTable } from "../../../components/tables/GenericTable";
import { ChakraModal } from "../../../components/chakra/modal";
import { useModal } from "../../../hooks/useOpen";
import { BUTTON_VARIANTS } from "../../../theme";
import { ConfirmInput } from "../../../components/form/confirm-input";
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
import { PythonIcon } from "../../../components/icons/python";
import { RunPythonOnAllCols } from "../../../components/RunPythonOnAllCols";
import { FormSubmitBar } from "../../../components/form/FormSubmitBar";
import { createPythonCode } from "../../../utils/str";
import { usePathParams } from "../../../hooks/usePathParams";
import { ConfirmModal } from "../../../components/form/confirm";
import {
  execPythonOnDataset,
  setTargetColumnReq,
} from "../../../clients/requests";
import { getValueById } from "../../../utils/dom";
import { ChakraPopover } from "../../../components/chakra/popover";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../../components/SelectFilter";
import { MultiValue, SingleValue } from "react-select";
import { CreateCopyPopover } from "../../../components/CreateCopyPopover";
import { SelectColumnPopover } from "../../../components/SelectTargetColumnPopover";
import { getDatasetColumnOptions } from "../../../utils/dataset";

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

  const dataset = data?.res.dataset;

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

  const setKlineOpenTimeColumn = async (klineOpenTimeColumn: string) => {
    const url = URLS.set_time_column(datasetName);
    const request = fetch(url, {
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        new_timeseries_col: klineOpenTimeColumn,
      }),
      method: "PUT",
    });
    request.then((res) => {
      if (res.status === 200) {
        toast({
          title: "Changed candle open time column",
          status: "info",
          duration: 5000,
          isClosable: true,
        });
        refetch();
        klineOpenTimePopover.onClose();
      }
    });
  };

  const setBacktestPriceColumn = async (value: string) => {
    const res = await setTargetColumnReq(datasetName, value);
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
        title={"Run python on all the columns"}
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
          <ChakraPopover
            isOpen={createCopyPopover.isOpen}
            setOpen={createCopyPopover.onOpen}
            onClose={createCopyPopover.onClose}
            headerText="Create a copy of the dataset"
            body={
              <CreateCopyPopover
                datasetName={datasetName}
                successCallback={() => {
                  createCopyPopover.onClose();
                }}
              />
            }
          >
            <Button variant={BUTTON_VARIANTS.grey}>Create copy</Button>
          </ChakraPopover>

          <ChakraPopover
            isOpen={backtestPriceColumnPopover.isOpen}
            setOpen={backtestPriceColumnPopover.onOpen}
            onClose={backtestPriceColumnPopover.onClose}
            headerText="Set backtest price column"
            body={
              <SelectColumnPopover
                options={getDatasetColumnOptions(data.res.dataset)}
                placeholder=""
                selectCallback={setBacktestPriceColumn}
              />
            }
          >
            <Button variant={BUTTON_VARIANTS.grey}>
              Set backtest price column
            </Button>
          </ChakraPopover>

          <ChakraPopover
            isOpen={klineOpenTimePopover.isOpen}
            setOpen={klineOpenTimePopover.onOpen}
            onClose={klineOpenTimePopover.onClose}
            headerText="Set candle open time column"
            body={
              <SelectColumnPopover
                options={getDatasetColumnOptions(data.res.dataset)}
                placeholder={data.res.dataset.timeseries_col}
                selectCallback={setKlineOpenTimeColumn}
              />
            }
          >
            <Button variant={BUTTON_VARIANTS.grey}>
              Set candle open time column
            </Button>
          </ChakraPopover>
          <ChakraPopover
            isOpen={targetColumnPopover.isOpen}
            setOpen={targetColumnPopover.onOpen}
            onClose={targetColumnPopover.onClose}
            headerText="Set target column"
            body={
              <SelectColumnPopover
                options={getDatasetColumnOptions(data.res.dataset)}
                placeholder={data.res.dataset.target_col}
                selectCallback={setTargetColumn}
              />
            }
          >
            <Button variant={BUTTON_VARIANTS.grey}>Set target column</Button>
          </ChakraPopover>
          <Button
            variant={BUTTON_VARIANTS.grey}
            leftIcon={<PythonIcon width={24} height={24} />}
            onClick={runPythonModal.modalOpen}
          >
            Run python
          </Button>
        </Box>
      </Box>
      <Box marginTop={"16px"}>
        <Box>Tail</Box>
        <GenericTable
          columns={columns}
          rows={dataset.tail}
          columnOnClickFunc={columnOnClickFunction}
        />
      </Box>
    </div>
  );
};
