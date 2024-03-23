import React, { useState } from "react";
import { CodeEditor } from "./CodeEditor";
import { CREATE_COLUMNS_DEFAULT } from "../utils/code";
import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Spinner,
  Switch,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import { WithLabel } from "./form/WithLabel";
import { ChakraModal } from "./chakra/modal";
import { Text } from "@chakra-ui/react";
import { TEXT_VARIANTS } from "../theme";
import { useDatasetQuery } from "../clients/queries/queries";
import { usePathParams } from "../hooks/usePathParams";
import { OverflopTooltip } from "./OverflowTooltip";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { execPythonOnDataset, updatePriceColumnReq } from "../clients/requests";
import { ChakraInput } from "./chakra/input";
import { ChakraPopover } from "./chakra/popover";
import { SelectColumnPopover } from "./SelectTargetColumnPopover";
import { getDatasetColumnOptions } from "../utils/dataset";
import { Dataset } from "../clients/queries/response-types";
import { CODE_PRESET_CATEGORY } from "../utils/constants";
import { ColumnInfoModal } from "./ColumnInfoModal";

type PathParams = {
  datasetName: string;
};

interface Props {
  openLongTradeCode: string;
  openShortTradeCode: string;
  closeLongTradeCode: string;
  closeShortTradeCode: string;
  setOpenLongTradeCode: React.Dispatch<React.SetStateAction<string>>;
  setOpenShortTradeCode: React.Dispatch<React.SetStateAction<string>>;
  setCloseLongTradeCode: React.Dispatch<React.SetStateAction<string>>;
  setCloseShortTradeCode: React.Dispatch<React.SetStateAction<string>>;
  useShorts: boolean;
  useProfitBasedClose: boolean;
  setUseProfitBasedClose: React.Dispatch<React.SetStateAction<boolean>>;
  setUseShorts: React.Dispatch<React.SetStateAction<boolean>>;
  backtestName: string;
  setBacktestName: React.Dispatch<React.SetStateAction<string>>;
  klinesUntilClose: null | number;
  takeProfitThresholdPerc: number;
  setTakeProfitThresholdPerc: React.Dispatch<React.SetStateAction<number>>;
  setKlinesUntilClose: React.Dispatch<React.SetStateAction<number | null>>;
  useTimeBasedClose: boolean;
  setUseTimeBasedClose: React.Dispatch<React.SetStateAction<boolean>>;
  tradingFees: number;
  setTradingFees: React.Dispatch<React.SetStateAction<number>>;
  slippage: number;
  setSlippage: React.Dispatch<React.SetStateAction<number>>;
  dataset: Dataset;
}

export const CreateBacktestDrawer = (props: Props) => {
  const {
    openLongTradeCode,
    openShortTradeCode,
    closeLongTradeCode,
    closeShortTradeCode,
    setOpenLongTradeCode,
    setOpenShortTradeCode,
    setCloseLongTradeCode,
    setCloseShortTradeCode,
    useShorts,
    setUseShorts,
    setBacktestName,
    useTimeBasedClose,
    setUseTimeBasedClose,
    klinesUntilClose,
    setKlinesUntilClose,
    tradingFees,
    setTradingFees,
    slippage,
    setSlippage,
    dataset,
  } = props;

  const { datasetName } = usePathParams<PathParams>();
  const [createColumnsCode, setCreateColumnsCode] = useState(
    CREATE_COLUMNS_DEFAULT()
  );

  const columnsModal = useDisclosure();
  const runPythonModal = useDisclosure();
  const columnDetailsModal = useDisclosure();
  const [selectedColumnName, setSelectedColumnName] = useState("");

  const { data, refetch: refetchDataset } = useDatasetQuery(datasetName);
  const toast = useToast();
  const backtestPriceColumnPopover = useDisclosure();

  const runPythonSubmit = async () => {
    const res = await execPythonOnDataset(
      datasetName,
      createColumnsCode,
      "NONE"
    );

    if (res.status === 200) {
      toast({
        title: "Executed python code",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      runPythonModal.onClose();
      refetchDataset();
      setCreateColumnsCode(CREATE_COLUMNS_DEFAULT());
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
      refetchDataset();
    }
  };

  if (!data) return <Spinner />;

  return (
    <div>
      <ChakraModal {...columnsModal} title="Columns">
        <div id={"COLUMN_MODAL"}>
          {data.columns.map((item, idx) => {
            return (
              <div
                key={idx}
                className="link-default"
                onClick={() => {
                  setSelectedColumnName(item);
                  columnDetailsModal.onOpen();
                }}
              >
                <OverflopTooltip text={item} containerId="COLUMN_MODAL">
                  <div>{item}</div>
                </OverflopTooltip>
              </div>
            );
          })}
        </div>
        {selectedColumnName && (
          <ChakraModal
            {...columnDetailsModal}
            title={`Column ${selectedColumnName}`}
            modalContentStyle={{ maxWidth: "80%", marginTop: "5%" }}
          >
            <ColumnInfoModal
              datasetName={datasetName}
              columnName={selectedColumnName}
            />
          </ChakraModal>
        )}
      </ChakraModal>

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

      <ChakraModal
        {...runPythonModal}
        title="Run python"
        footerContent={
          <FormSubmitBar
            cancelCallback={runPythonModal.onClose}
            submitCallback={runPythonSubmit}
          />
        }
        modalContentStyle={{ maxWidth: "60%" }}
      >
        <CodeEditor
          code={createColumnsCode}
          setCode={setCreateColumnsCode}
          style={{ marginTop: "16px" }}
          fontSize={13}
          label="Create columns"
          disableCodePresets={true}
          codeContainerStyles={{ width: "100%" }}
          height={"250px"}
          presetCategory={CODE_PRESET_CATEGORY.backtest_create_columns}
        />
      </ChakraModal>

      <div style={{ display: "flex", gap: "16px" }}>
        <Text variant={TEXT_VARIANTS.clickable} onClick={columnsModal.onOpen}>
          Show columns
        </Text>
        <Text variant={TEXT_VARIANTS.clickable} onClick={runPythonModal.onOpen}>
          Run python
        </Text>
        <Text
          variant={TEXT_VARIANTS.clickable}
          onClick={backtestPriceColumnPopover.onOpen}
        >
          Set price column
        </Text>
      </div>

      <WithLabel>
        <ChakraInput label="Name (optional)" onChange={setBacktestName} />
      </WithLabel>
      <div>
        <CodeEditor
          code={openLongTradeCode}
          setCode={setOpenLongTradeCode}
          style={{ marginTop: "16px" }}
          fontSize={13}
          label={"Long condition"}
          disableCodePresets={true}
          codeContainerStyles={{ width: "100%" }}
          height={"250px"}
          presetCategory={CODE_PRESET_CATEGORY.backtest_long_cond}
        />
      </div>

      <div>
        <CodeEditor
          code={closeLongTradeCode}
          setCode={setCloseLongTradeCode}
          style={{ marginTop: "16px" }}
          fontSize={13}
          label="Close long condition"
          disableCodePresets={true}
          codeContainerStyles={{ width: "100%" }}
          height={"250px"}
          presetCategory={CODE_PRESET_CATEGORY.backtest_close_long_ccond}
        />
      </div>

      <div style={{ marginTop: "16px" }}>
        <WithLabel label={"Use short selling"}>
          <Switch
            isChecked={useShorts}
            onChange={() => setUseShorts(!useShorts)}
          />
        </WithLabel>
      </div>

      {useShorts && (
        <div style={{ marginTop: "16px" }}>
          <CodeEditor
            code={openShortTradeCode}
            setCode={setOpenShortTradeCode}
            style={{ marginTop: "16px" }}
            fontSize={13}
            label="Short condition"
            disableCodePresets={true}
            codeContainerStyles={{ width: "100%" }}
            height={"250px"}
          />
        </div>
      )}

      {useShorts && (
        <div style={{ marginTop: "16px" }}>
          <CodeEditor
            code={closeShortTradeCode}
            setCode={setCloseShortTradeCode}
            style={{ marginTop: "16px" }}
            fontSize={13}
            label="Close short condition"
            disableCodePresets={true}
            codeContainerStyles={{ width: "100%" }}
            height={"250px"}
          />
        </div>
      )}

      <div style={{ marginTop: "16px" }}>
        <WithLabel label={"Use time based closing strategy"}>
          <Switch
            isChecked={useTimeBasedClose}
            onChange={() => setUseTimeBasedClose(!useTimeBasedClose)}
          />
        </WithLabel>
      </div>

      {useTimeBasedClose && (
        <WithLabel
          label={"Klines until close"}
          containerStyles={{ maxWidth: "200px", marginTop: "16px" }}
        >
          <NumberInput
            step={5}
            min={0}
            value={klinesUntilClose || undefined}
            onChange={(valueString) =>
              setKlinesUntilClose(parseInt(valueString))
            }
          >
            <NumberInputField />
          </NumberInput>
        </WithLabel>
      )}

      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <WithLabel
          label={"Trading fees (%)"}
          containerStyles={{ maxWidth: "200px", marginTop: "16px" }}
        >
          <NumberInput
            step={0.005}
            min={0}
            value={tradingFees}
            precision={3}
            onChange={(valueString) => setTradingFees(parseFloat(valueString))}
          >
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </WithLabel>
        <WithLabel
          label={"Slippage (%)"}
          containerStyles={{ maxWidth: "200px", marginTop: "16px" }}
        >
          <NumberInput
            step={0.001}
            min={0}
            value={slippage}
            precision={4}
            onChange={(valueString) => setSlippage(parseFloat(valueString))}
          >
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper color={"white"} />
              <NumberDecrementStepper color={"white"} />
            </NumberInputStepper>
          </NumberInput>
        </WithLabel>
      </div>
    </div>
  );
};
