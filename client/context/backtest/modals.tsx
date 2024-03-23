import React, { useState } from "react";
import { useBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";
import {
  CREATE_COLUMNS_DEFAULT,
  ENTER_TRADE_DEFAULT,
  EXIT_LONG_TRADE_DEFAULT,
  EXIT_SHORT_TRADE_DEFAULT,
  EXIT_TRADE_DEFAULT,
} from "../../utils/code";
import {
  createManualBacktest,
  execPythonOnDataset,
  updatePriceColumnReq,
} from "../../clients/requests";
import { usePathParams } from "../../hooks/usePathParams";
import { useDatasetQuery } from "../../clients/queries/queries";
import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Spinner,
  Switch,
  Text,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import { ChakraModal } from "../../components/chakra/modal";
import { OverflopTooltip } from "../../components/OverflowTooltip";
import { ColumnInfoModal } from "../../components/ColumnInfoModal";
import { ChakraPopover } from "../../components/chakra/popover";
import { SelectColumnPopover } from "../../components/SelectTargetColumnPopover";
import { getDatasetColumnOptions } from "../../utils/dataset";
import { CodeEditor } from "../../components/CodeEditor";
import { CODE_PRESET_CATEGORY } from "../../utils/constants";
import { TEXT_VARIANTS } from "../../theme";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";

type PathParams = {
  datasetName: string;
};

export const BacktestUXManager = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { data: dataset } = useDatasetQuery(datasetName);

  const { createNewDrawer, datasetBacktestsQuery, forceUpdate } =
    useBacktestContext();

  const [openLongTradeCode, setOpenLongTradeCode] = useState(
    ENTER_TRADE_DEFAULT()
  );
  const [openShortTradeCode, setOpenShortTradeCode] =
    useState(EXIT_TRADE_DEFAULT());

  const [closeLongTradeCode, setCloseLongTradeCode] = useState(
    EXIT_LONG_TRADE_DEFAULT()
  );
  const [closeShortTradeCode, setCloseShortTradeCode] = useState(
    EXIT_SHORT_TRADE_DEFAULT()
  );
  const [backtestName, setBacktestName] = useState("");
  const [useShorts, setUseShorts] = useState(false);

  const [useTimeBasedClose, setUseTimeBasedClose] = useState(false);
  const [useProfitBasedClose, setUseProfitBasedClose] = useState(false);
  const [useStopLossBasedClose, setUseStopLossBasedClose] = useState(false);
  const [klinesUntilClose, setKlinesUntilClose] = useState<null | number>(null);
  const [tradingFees, setTradingFees] = useState(0.1);
  const [slippage, setSlippage] = useState(0.001);
  const [shortFeeHourly, setShortFeeHourly] = useState(0.001);
  const [takeProfitThresholdPerc, setTakeProfitThresholdPerc] = useState(0);
  const [stopLossThresholdPerc, setStopLossThresholdPerc] = useState(0);

  const toast = useToast();

  const [createColumnsCode, setCreateColumnsCode] = useState(
    CREATE_COLUMNS_DEFAULT()
  );

  const columnsModal = useDisclosure();
  const runPythonModal = useDisclosure();
  const columnDetailsModal = useDisclosure();
  const [selectedColumnName, setSelectedColumnName] = useState("");

  const { data, refetch: refetchDataset } = useDatasetQuery(datasetName);
  const backtestPriceColumnPopover = useDisclosure();

  const submitNewBacktest = async () => {
    if (!dataset) return;

    const res = await createManualBacktest({
      open_long_trade_cond: openLongTradeCode,
      close_long_trade_cond: closeLongTradeCode,
      open_short_trade_cond: openShortTradeCode,
      close_short_trade_cond: closeShortTradeCode,
      use_short_selling: useShorts,
      dataset_id: dataset.id,
      name: backtestName,
      use_time_based_close: useTimeBasedClose,
      klines_until_close: klinesUntilClose,
      trading_fees_perc: tradingFees,
      slippage_perc: slippage,
    });

    if (res.status === 200) {
      toast({
        title: "Finished backtest",
        description: `Result: ${
          res.res.data.end_balance - res.res.data.start_balance
        }`,
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      createNewDrawer.onClose();
      datasetBacktestsQuery.refetch();
      forceUpdate();
    }
  };

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

  if (!dataset || !data) return <Spinner />;

  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...createNewDrawer}
        footerContent={
          <FormSubmitBar
            submitCallback={submitNewBacktest}
            cancelCallback={createNewDrawer.onClose}
          />
        }
      >
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
            <Text
              variant={TEXT_VARIANTS.clickable}
              onClick={columnsModal.onOpen}
            >
              Show columns
            </Text>
            <Text
              variant={TEXT_VARIANTS.clickable}
              onClick={runPythonModal.onOpen}
            >
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
                onChange={(valueString) =>
                  setTradingFees(parseFloat(valueString))
                }
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
      </ChakraDrawer>
    </div>
  );
};
