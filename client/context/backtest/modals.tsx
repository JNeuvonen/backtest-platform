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
  Button,
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
import { BUTTON_VARIANTS, TEXT_VARIANTS } from "../../theme";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";
import { Field, Formik, Form } from "formik";

type PathParams = {
  datasetName: string;
};

interface FormValues {
  backtestName: string;
  openLongTradeCode: string;
  closeLongTradeCode: string;
  openShortTradeCode: string;
  closeShortTradeCode: string;
  useShorts: boolean;
  useTimeBasedClose: boolean;
  useProfitBasedClose: boolean;
  useStopLossBasedClose: boolean;
  klinesUntilClose: number;
  tradingFees: number;
  slippage: number;
  shortFeeHourly: number;
  stopLossThresholdPerc: number;
  takeProfitThresholdPerc: number;
}

export const BacktestUXManager = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { data: dataset } = useDatasetQuery(datasetName);

  const { createNewDrawer, datasetBacktestsQuery, forceUpdate } =
    useBacktestContext();

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

  const submitNewBacktest = async (values: FormValues) => {
    if (!dataset) return;

    const res = await createManualBacktest({
      open_long_trade_cond: values.openLongTradeCode,
      close_long_trade_cond: values.closeLongTradeCode,
      open_short_trade_cond: values.openShortTradeCode,
      close_short_trade_cond: values.closeShortTradeCode,
      use_short_selling: values.useShorts,
      dataset_id: dataset.id,
      name: values.backtestName,
      use_time_based_close: values.useTimeBasedClose,
      klines_until_close: values.klinesUntilClose,
      trading_fees_perc: values.tradingFees,
      slippage_perc: values.slippage,
      short_fee_hourly: values.shortFeeHourly,
      use_stop_loss_based_close: values.useStopLossBasedClose,
      use_profit_based_close: values.useProfitBasedClose,
      stop_loss_threshold_perc: values.stopLossThresholdPerc,
      take_profit_threshold_perc: values.takeProfitThresholdPerc,
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

          <Formik
            onSubmit={(values) => {
              submitNewBacktest(values);
            }}
            initialValues={{
              backtestName: "",
              openLongTradeCode: ENTER_TRADE_DEFAULT(),
              closeLongTradeCode: EXIT_LONG_TRADE_DEFAULT(),
              openShortTradeCode: EXIT_TRADE_DEFAULT(),
              closeShortTradeCode: EXIT_SHORT_TRADE_DEFAULT(),
              useShorts: false,
              useTimeBasedClose: false,
              useProfitBasedClose: false,
              useStopLossBasedClose: false,
              klinesUntilClose: 0,
              tradingFees: 0.1,
              slippage: 0.001,
              shortFeeHourly: 0.001,
              takeProfitThresholdPerc: 0,
              stopLossThresholdPerc: 0,
            }}
          >
            {({ values }) => (
              <Form>
                <Field name="backtestName">
                  {({ form }) => {
                    return (
                      <WithLabel>
                        <ChakraInput
                          label="Name (optional)"
                          onChange={(value: string) =>
                            form.setFieldValue("backtestName", value)
                          }
                        />
                      </WithLabel>
                    );
                  }}
                </Field>
                <div>
                  <Field name="openLongTradeCode">
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue("openLongTradeCode", newState)
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={"Long condition"}
                          disableCodePresets={true}
                          codeContainerStyles={{ width: "100%" }}
                          height={"250px"}
                          presetCategory={
                            CODE_PRESET_CATEGORY.backtest_long_cond
                          }
                        />
                      );
                    }}
                  </Field>
                </div>

                <div>
                  <Field name="closeLongTradeCode">
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue("closeLongTradeCode", newState)
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label="Close long condition"
                          disableCodePresets={true}
                          codeContainerStyles={{ width: "100%" }}
                          height={"250px"}
                          presetCategory={
                            CODE_PRESET_CATEGORY.backtest_close_long_ccond
                          }
                        />
                      );
                    }}
                  </Field>
                </div>

                <div style={{ marginTop: "16px" }}>
                  <Field name="useShorts">
                    {({ field, form }) => {
                      return (
                        <WithLabel label={"Use short selling"}>
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue("useShorts", !field.value)
                            }
                          />
                        </WithLabel>
                      );
                    }}
                  </Field>
                </div>

                {values.useShorts && (
                  <div style={{ marginTop: "16px" }}>
                    <Field name="openShortTradeCode">
                      {({ field, form }) => {
                        return (
                          <CodeEditor
                            code={field.value}
                            setCode={(newState) =>
                              form.setFieldValue("openShortTradeCode", newState)
                            }
                            style={{ marginTop: "16px" }}
                            fontSize={13}
                            label="Short condition"
                            disableCodePresets={true}
                            codeContainerStyles={{ width: "100%" }}
                            height={"250px"}
                          />
                        );
                      }}
                    </Field>
                  </div>
                )}

                {values.useShorts && (
                  <div style={{ marginTop: "16px" }}>
                    <Field name="closeShortTradeCode">
                      {({ field, form }) => {
                        return (
                          <CodeEditor
                            code={field.value}
                            setCode={(newState) =>
                              form.setFieldValue(
                                "closeShortTradeCode",
                                newState
                              )
                            }
                            style={{ marginTop: "16px" }}
                            fontSize={13}
                            label="Close short condition"
                            disableCodePresets={true}
                            codeContainerStyles={{ width: "100%" }}
                            height={"250px"}
                          />
                        );
                      }}
                    </Field>
                  </div>
                )}

                <div style={{ marginTop: "16px" }}>
                  <Field name="useTimeBasedClose">
                    {({ field, form }) => {
                      return (
                        <WithLabel label={"Use time based closing strategy"}>
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue(
                                "useTimeBasedClose",
                                !field.value
                              )
                            }
                          />
                        </WithLabel>
                      );
                    }}
                  </Field>
                </div>

                {values.useTimeBasedClose && (
                  <Field name="klinesUntilClose">
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Klines until close"}
                          containerStyles={{
                            maxWidth: "200px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={5}
                            min={0}
                            value={field.value}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                "klinesUntilClose",
                                parseInt(valueString)
                              )
                            }
                          >
                            <NumberInputField />
                          </NumberInput>
                        </WithLabel>
                      );
                    }}
                  </Field>
                )}

                <div style={{ marginTop: "16px" }}>
                  <Field name="useProfitBasedClose">
                    {({ field, form }) => {
                      return (
                        <WithLabel label={"Use profit based close"}>
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue(
                                "useProfitBasedClose",
                                !field.value
                              )
                            }
                          />
                        </WithLabel>
                      );
                    }}
                  </Field>
                </div>

                {values.useProfitBasedClose && (
                  <Field name="takeProfitThresholdPerc">
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Take profit threshold (%)"}
                          containerStyles={{
                            maxWidth: "200px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={5}
                            min={0}
                            value={field.value}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                "takeProfitThresholdPerc",
                                parseInt(valueString)
                              )
                            }
                          >
                            <NumberInputField />
                          </NumberInput>
                        </WithLabel>
                      );
                    }}
                  </Field>
                )}

                <div style={{ marginTop: "16px" }}>
                  <Field name="useStopLossBasedClose">
                    {({ field, form }) => {
                      return (
                        <WithLabel label={"Use stop loss based close"}>
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue(
                                "useStopLossBasedClose",
                                !field.value
                              )
                            }
                          />
                        </WithLabel>
                      );
                    }}
                  </Field>
                </div>

                {values.useStopLossBasedClose && (
                  <Field name="stopLossThresholdPerc">
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Stop loss threshold (%)"}
                          containerStyles={{
                            maxWidth: "200px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={5}
                            min={0}
                            value={field.value}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                "stopLossThresholdPerc",
                                parseInt(valueString)
                              )
                            }
                          >
                            <NumberInputField />
                          </NumberInput>
                        </WithLabel>
                      );
                    }}
                  </Field>
                )}

                <div
                  style={{ display: "flex", alignItems: "center", gap: "16px" }}
                >
                  <Field name="tradingFees">
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Trading fees (%)"}
                          containerStyles={{
                            maxWidth: "200px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={0.005}
                            min={0}
                            value={field.value}
                            precision={3}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                "tradingFees",
                                parseFloat(valueString)
                              )
                            }
                          >
                            <NumberInputField />
                            <NumberInputStepper>
                              <NumberIncrementStepper />
                              <NumberDecrementStepper />
                            </NumberInputStepper>
                          </NumberInput>
                        </WithLabel>
                      );
                    }}
                  </Field>

                  <Field name="slippage">
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Slippage (%)"}
                          containerStyles={{
                            maxWidth: "200px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={0.001}
                            min={0}
                            value={field.value}
                            precision={4}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                "slippage",
                                parseFloat(valueString)
                              )
                            }
                          >
                            <NumberInputField />
                            <NumberInputStepper>
                              <NumberIncrementStepper color={"white"} />
                              <NumberDecrementStepper color={"white"} />
                            </NumberInputStepper>
                          </NumberInput>
                        </WithLabel>
                      );
                    }}
                  </Field>

                  {values.useShorts && (
                    <Field name="shortFeeHourly">
                      {({ field, form }) => {
                        return (
                          <WithLabel
                            label={"Shorting fees (%) hourly"}
                            containerStyles={{
                              maxWidth: "200px",
                              marginTop: "16px",
                            }}
                          >
                            <NumberInput
                              step={0.005}
                              min={0}
                              value={field.value}
                              precision={3}
                              onChange={(valueString) =>
                                form.setFieldValue(
                                  "shortFeeHourly",
                                  parseFloat(valueString)
                                )
                              }
                            >
                              <NumberInputField />
                              <NumberInputStepper>
                                <NumberIncrementStepper />
                                <NumberDecrementStepper />
                              </NumberInputStepper>
                            </NumberInput>
                          </WithLabel>
                        );
                      }}
                    </Field>
                  )}
                </div>

                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginTop: "16px",
                  }}
                >
                  <Button
                    variant={BUTTON_VARIANTS.nofill}
                    onClick={createNewDrawer.onClose}
                  >
                    Cancel
                  </Button>
                  <Button type="submit" marginTop={"16px"}>
                    Run backtest
                  </Button>
                </div>
              </Form>
            )}
          </Formik>
        </div>
      </ChakraDrawer>
    </div>
  );
};
