import React from "react";
import { ChakraDrawer } from "../../../../components/chakra/Drawer";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useBacktestById } from "../../../../clients/queries/queries";
import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Switch,
  UseDisclosureReturn,
} from "@chakra-ui/react";
import { FormSubmitBar } from "../../../../components/form/FormSubmitBar";
import { Field, Form, Formik, FormikProps } from "formik";
import { BacktestObject } from "../../../../clients/queries/response-types";
import { DISK_KEYS, DiskManager } from "../../../../utils/disk";
import { WithLabel } from "../../../../components/form/WithLabel";
import { ChakraInput } from "../../../../components/chakra/input";
import { CodeEditor } from "../../../../components/CodeEditor";
import { CODE_PRESET_CATEGORY } from "../../../../utils/constants";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

interface Props {
  deployStrategyDrawer: UseDisclosureReturn;
}

const backtestDiskManager = new DiskManager(DISK_KEYS.deploy_strategy_form);

const formKeys = {
  name: "name",
  symbol: "symbol",
  baseAsset: "base_asset",
  quoteAsset: "quote_asset",
  enterTradeCode: "enter_trade_code",
  exitTradeCode: "exit_trade_code",
  fetchDatasourcesCode: "fetch_datasources_code",
  dataTransformationsCode: "data_transformations_code",
  tradeQuantityPrecision: "trade_quantity_precision",
  priority: "priority",
  klineSizeMs: "kline_size_ms",
  maximumKlinesHoldTime: "maximum_klines_hold_time",
  allocatedSizePerc: "allocated_size_perc",
  takeProfitThresholdPerc: "take_profit_threshold_perc",
  stopLossThresholdPerc: "stop_loss_threshold_perc",
  minimumTimeBetweenTradesMs: "minimum_time_between_trades_ms",
  useTimeBasedClose: "use_time_based_close",
  useProfitBasedClose: "use_profit_based_Close",
  useStopLossBasedClose: "use_stop_loss_based_close",
  useTakerOrder: "use_taker_order",
  isLeverageAllowed: "is_leverage_allowed",
  isShortSellingStrategy: "is_short_selling_strategy",
  isPaperTradeMode: "is_paper_trade_mode",
};

interface FormValues {
  name: string;
  symbol: string;
  base_asset: string;
  quote_asset: string;
  enter_trade_code: string;
  exit_trade_code: string;
  fetch_datasources_code: string;
  data_transformations_code: string;
  trade_quantity_precision: number;
  priority: number;
  kline_size_ms: number;
  maximum_klines_hold_time: number;
  allocated_size_perc: number;
  take_profit_threshold_perc: number;
  stop_loss_threshold_perc: number;
  minimum_time_between_trades_ms: number;
  use_time_based_close: boolean;
  use_profit_based_Close: boolean;
  use_stop_loss_based_close: boolean;
  use_taker_order: boolean;
  is_leverage_allowed: boolean;
  is_short_selling_strategy: boolean;
  is_paper_trade_mode: boolean;
}

const getFormInitialValues = (backtest: BacktestObject): FormValues => {
  const prevForm = backtestDiskManager.read();
  if (prevForm === null) {
    return {
      name: "",
      symbol: "",
      base_asset: "",
      quote_asset: "",
      enter_trade_code: "",
      exit_trade_code: "",
      fetch_datasources_code: "",
      data_transformations_code: "",
      trade_quantity_precision: 5,
      priority: 1,
      kline_size_ms: 0,
      maximum_klines_hold_time: backtest.klines_until_close,
      allocated_size_perc: 25,
      take_profit_threshold_perc: backtest.take_profit_threshold_perc,
      stop_loss_threshold_perc: backtest.stop_loss_threshold_perc,
      minimum_time_between_trades_ms: 0,
      use_time_based_close: false,
      use_profit_based_Close: false,
      use_stop_loss_based_close: backtest.use_stop_loss_based_close,
      use_taker_order: false,
      is_leverage_allowed: false,
      is_short_selling_strategy: false,
      is_paper_trade_mode: false,
    };
  }

  return {
    ...prevForm,
    name: "",
  };
};

const onSubmit = () => {};

export const DeployStrategyForm = (props: Props) => {
  const { deployStrategyDrawer } = props;
  const { backtestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));

  if (!backtestQuery || !backtestQuery.data || !backtestQuery.data.data) {
    return null;
  }

  const backtest = backtestQuery.data.data;

  return (
    <div>
      <ChakraDrawer
        {...deployStrategyDrawer}
        title={`Deploy strategy ${backtestQuery.data?.data.name}`}
        drawerContentStyles={{ maxWidth: "80%" }}
      >
        <div>
          <Formik
            initialValues={getFormInitialValues(backtest)}
            onSubmit={onSubmit}
          >
            {({ values }: FormikProps<FormValues>) => {
              return (
                <Form>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "16px",
                    }}
                  >
                    <div>
                      <Field name={formKeys.name}>
                        {({ form }) => {
                          return (
                            <WithLabel>
                              <ChakraInput
                                label="Strategy name"
                                onChange={(value: string) =>
                                  form.setFieldValue(formKeys.name, value)
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.symbol}>
                        {({ form }) => {
                          return (
                            <WithLabel>
                              <ChakraInput
                                label="Symbol"
                                onChange={(value: string) =>
                                  form.setFieldValue(formKeys.symbol, value)
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.baseAsset}>
                        {({ form }) => {
                          return (
                            <WithLabel>
                              <ChakraInput
                                label="Base asset"
                                onChange={(value: string) =>
                                  form.setFieldValue(formKeys.baseAsset, value)
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.quoteAsset}>
                        {({ form }) => {
                          return (
                            <WithLabel>
                              <ChakraInput
                                label="Quote asset"
                                onChange={(value: string) =>
                                  form.setFieldValue(formKeys.baseAsset, value)
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                  </div>
                  <div>
                    <Field name={formKeys.enterTradeCode}>
                      {({ field, form }) => {
                        return (
                          <CodeEditor
                            code={field.value}
                            setCode={(newState) =>
                              form.setFieldValue(
                                formKeys.enterTradeCode,
                                newState
                              )
                            }
                            style={{ marginTop: "16px" }}
                            fontSize={13}
                            label={"Enter trade code"}
                            codeContainerStyles={{ width: "100%" }}
                            height={"250px"}
                            presetCategory={
                              CODE_PRESET_CATEGORY.strategy_deploy_enter_trade
                            }
                          />
                        );
                      }}
                    </Field>
                  </div>
                  <div>
                    <Field name={formKeys.exitTradeCode}>
                      {({ field, form }) => {
                        return (
                          <CodeEditor
                            code={field.value}
                            setCode={(newState) =>
                              form.setFieldValue(
                                formKeys.exitTradeCode,
                                newState
                              )
                            }
                            style={{ marginTop: "16px" }}
                            fontSize={13}
                            label={"Exit trade code"}
                            codeContainerStyles={{ width: "100%" }}
                            height={"250px"}
                            presetCategory={
                              CODE_PRESET_CATEGORY.strategy_deploy_exit_trade
                            }
                          />
                        );
                      }}
                    </Field>
                  </div>
                  <div>
                    <Field name={formKeys.fetchDatasourcesCode}>
                      {({ field, form }) => {
                        return (
                          <CodeEditor
                            code={field.value}
                            setCode={(newState) =>
                              form.setFieldValue(
                                formKeys.fetchDatasourcesCode,
                                newState
                              )
                            }
                            style={{ marginTop: "16px" }}
                            fontSize={13}
                            label={"Fetch datasources code"}
                            codeContainerStyles={{ width: "100%" }}
                            height={"250px"}
                            presetCategory={
                              CODE_PRESET_CATEGORY.strategy_deploy_fetch_datasources
                            }
                          />
                        );
                      }}
                    </Field>
                  </div>
                  <div>
                    <Field name={formKeys.dataTransformationsCode}>
                      {({ field, form }) => {
                        return (
                          <CodeEditor
                            code={field.value}
                            setCode={(newState) =>
                              form.setFieldValue(
                                formKeys.dataTransformationsCode,
                                newState
                              )
                            }
                            style={{ marginTop: "16px" }}
                            fontSize={13}
                            label={"Transform data code"}
                            codeContainerStyles={{ width: "100%" }}
                            height={"250px"}
                            presetCategory={
                              CODE_PRESET_CATEGORY.strategy_deploy_data_transformations
                            }
                          />
                        );
                      }}
                    </Field>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "16px",
                      marginTop: "16px",
                    }}
                  >
                    <div>
                      <Field name={formKeys.tradeQuantityPrecision}>
                        {({ field, form }) => {
                          return (
                            <WithLabel
                              label={"Trade quantity precision"}
                              containerStyles={{
                                maxWidth: "200px",
                                marginTop: "16px",
                              }}
                            >
                              <NumberInput
                                step={1}
                                min={0}
                                value={field.value}
                                onChange={(valueString) =>
                                  form.setFieldValue(
                                    formKeys.tradeQuantityPrecision,
                                    parseInt(valueString)
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
                    </div>
                    <div>
                      <Field name={formKeys.priority}>
                        {({ field, form }) => {
                          return (
                            <WithLabel
                              label={"Priority"}
                              containerStyles={{
                                maxWidth: "200px",
                                marginTop: "16px",
                              }}
                            >
                              <NumberInput
                                step={1}
                                min={0}
                                value={field.value}
                                onChange={(valueString) =>
                                  form.setFieldValue(
                                    formKeys.priority,
                                    parseInt(valueString)
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
                    </div>
                    <div>
                      <Field name={formKeys.klineSizeMs}>
                        {({ field, form }) => {
                          return (
                            <WithLabel
                              label={"Kline size MS"}
                              containerStyles={{
                                maxWidth: "200px",
                                marginTop: "16px",
                              }}
                            >
                              <NumberInput
                                step={10000}
                                min={0}
                                value={field.value}
                                onChange={(valueString) =>
                                  form.setFieldValue(
                                    formKeys.klineSizeMs,
                                    parseInt(valueString)
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
                    </div>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "16px",
                      marginTop: "16px",
                    }}
                  >
                    <div>
                      <Field name={formKeys.maximumKlinesHoldTime}>
                        {({ field, form }) => {
                          return (
                            <WithLabel
                              label={"Hold time in klines"}
                              containerStyles={{
                                maxWidth: "200px",
                              }}
                            >
                              <NumberInput
                                step={5}
                                min={0}
                                value={field.value}
                                onChange={(valueString) =>
                                  form.setFieldValue(
                                    formKeys.maximumKlinesHoldTime,
                                    parseInt(valueString)
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
                    </div>
                    <div>
                      <Field name={formKeys.allocatedSizePerc}>
                        {({ field, form }) => {
                          return (
                            <WithLabel
                              label={"Allocated size perc"}
                              containerStyles={{
                                maxWidth: "200px",
                              }}
                            >
                              <NumberInput
                                step={1}
                                min={0}
                                value={field.value}
                                onChange={(valueString) =>
                                  form.setFieldValue(
                                    formKeys.allocatedSizePerc,
                                    parseInt(valueString)
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
                    </div>

                    <div>
                      <Field name={formKeys.takeProfitThresholdPerc}>
                        {({ field, form }) => {
                          return (
                            <WithLabel
                              label={"Take profit threshold (%)"}
                              containerStyles={{
                                maxWidth: "200px",
                              }}
                            >
                              <NumberInput
                                step={1}
                                min={0}
                                value={field.value}
                                onChange={(valueString) =>
                                  form.setFieldValue(
                                    formKeys.takeProfitThresholdPerc,
                                    parseInt(valueString)
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
                    </div>

                    <div>
                      <Field name={formKeys.minimumTimeBetweenTradesMs}>
                        {({ field, form }) => {
                          return (
                            <WithLabel
                              label={"Time between trades MS"}
                              containerStyles={{
                                maxWidth: "200px",
                              }}
                            >
                              <NumberInput
                                step={5000}
                                min={0}
                                value={field.value}
                                onChange={(valueString) =>
                                  form.setFieldValue(
                                    formKeys.minimumTimeBetweenTradesMs,
                                    parseInt(valueString)
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
                    </div>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "16px",
                      marginTop: "16px",
                    }}
                  >
                    <div>
                      <Field name={formKeys.useTimeBasedClose}>
                        {({ field, form }) => {
                          return (
                            <WithLabel label={"Use time based close"}>
                              <Switch
                                isChecked={field.value}
                                onChange={() =>
                                  form.setFieldValue(
                                    formKeys.useTimeBasedClose,
                                    !field.value
                                  )
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.useStopLossBasedClose}>
                        {({ field, form }) => {
                          return (
                            <WithLabel label={"Use stop loss based close (%)"}>
                              <Switch
                                isChecked={field.value}
                                onChange={() =>
                                  form.setFieldValue(
                                    formKeys.useStopLossBasedClose,
                                    !field.value
                                  )
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.useProfitBasedClose}>
                        {({ field, form }) => {
                          return (
                            <WithLabel label={"Use profit based close (%)"}>
                              <Switch
                                isChecked={field.value}
                                onChange={() =>
                                  form.setFieldValue(
                                    formKeys.useProfitBasedClose,
                                    !field.value
                                  )
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.useTakerOrder}>
                        {({ field, form }) => {
                          return (
                            <WithLabel label={"Use market orders"}>
                              <Switch
                                isChecked={field.value}
                                onChange={() =>
                                  form.setFieldValue(
                                    formKeys.useTakerOrder,
                                    !field.value
                                  )
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.isLeverageAllowed}>
                        {({ field, form }) => {
                          return (
                            <WithLabel label={"Allow leverage on longs"}>
                              <Switch
                                isChecked={field.value}
                                onChange={() =>
                                  form.setFieldValue(
                                    formKeys.isLeverageAllowed,
                                    !field.value
                                  )
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                  </div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "16px",
                      marginTop: "16px",
                    }}
                  >
                    <div>
                      <Field name={formKeys.isShortSellingStrategy}>
                        {({ field, form }) => {
                          return (
                            <WithLabel label={"Short selling strategy"}>
                              <Switch
                                isChecked={field.value}
                                onChange={() =>
                                  form.setFieldValue(
                                    formKeys.isShortSellingStrategy,
                                    !field.value
                                  )
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.isPaperTradeMode}>
                        {({ field, form }) => {
                          return (
                            <WithLabel label={"Paper trade mode"}>
                              <Switch
                                isChecked={field.value}
                                onChange={() =>
                                  form.setFieldValue(
                                    formKeys.isPaperTradeMode,
                                    !field.value
                                  )
                                }
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                  </div>
                </Form>
              );
            }}
          </Formik>
          <FormSubmitBar style={{ marginTop: "16px" }} />
        </div>
      </ChakraDrawer>
    </div>
  );
};
