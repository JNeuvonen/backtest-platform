import React from "react";
import { ChakraDrawer } from "../../../../components/chakra/Drawer";
import { usePathParams } from "../../../../hooks/usePathParams";
import {
  useBacktestById,
  useDatasetQuery,
} from "../../../../clients/queries/queries";
import {
  Button,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Spinner,
  Switch,
  UseDisclosureReturn,
  useToast,
} from "@chakra-ui/react";
import { Field, Form, Formik } from "formik";
import {
  BacktestObject,
  DataTransformation,
} from "../../../../clients/queries/response-types";
import { DISK_KEYS, DiskManager } from "../../../../utils/disk";
import { WithLabel } from "../../../../components/form/WithLabel";
import { ChakraInput } from "../../../../components/chakra/input";
import { CodeEditor } from "../../../../components/CodeEditor";
import { CODE_PRESET_CATEGORY } from "../../../../utils/constants";
import { BUTTON_VARIANTS } from "../../../../theme";
import { deployStrategyReq } from "../../../../clients/requests";
import { useAppContext } from "../../../../context/app";
import { FETCH_DATASOURCES_DEFAULT } from "../../../../utils/code";

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

export interface DeployStratForm {
  name: string;
  symbol: string;
  base_asset: string;
  quote_asset: string;
  enter_trade_code: string;
  exit_trade_code: string;
  fetch_datasources_code: string;
  trade_quantity_precision: number;
  priority: number;
  kline_size_ms: number;
  maximum_klines_hold_time: number;
  allocated_size_perc: number;
  take_profit_threshold_perc: number;
  stop_loss_threshold_perc: number;
  minimum_time_between_trades_ms: number;
  use_time_based_close: boolean;
  use_profit_based_close: boolean;
  use_stop_loss_based_close: boolean;
  use_taker_order: boolean;
  is_leverage_allowed: boolean;
  is_short_selling_strategy: boolean;
  is_paper_trade_mode: boolean;
  data_transformations: DataTransformation[];
}

const getFormInitialValues = (backtest: BacktestObject): DeployStratForm => {
  const prevForm = backtestDiskManager.read();
  if (prevForm === null) {
    return {
      name: "",
      symbol: "",
      base_asset: "",
      quote_asset: "",
      enter_trade_code: backtest.open_trade_cond,
      exit_trade_code: backtest.close_trade_cond,
      fetch_datasources_code: FETCH_DATASOURCES_DEFAULT,
      trade_quantity_precision: 5,
      priority: 1,
      kline_size_ms: 0,
      maximum_klines_hold_time: backtest.klines_until_close,
      allocated_size_perc: 25,
      take_profit_threshold_perc: backtest.take_profit_threshold_perc,
      stop_loss_threshold_perc: backtest.stop_loss_threshold_perc,
      minimum_time_between_trades_ms: 0,
      use_time_based_close: backtest.use_time_based_close,
      use_profit_based_close: backtest.use_profit_based_close,
      use_stop_loss_based_close: backtest.use_stop_loss_based_close,
      use_taker_order: true,
      is_leverage_allowed: false,
      is_short_selling_strategy: backtest.is_short_selling_strategy,
      is_paper_trade_mode: false,
      data_transformations: [],
    };
  }

  return {
    ...prevForm,
    enter_trade_code: backtest.open_trade_cond,
    exit_trade_code: backtest.close_trade_cond,
    is_short_selling_strategy: backtest.is_short_selling_strategy,
    maximum_klines_hold_time: backtest.klines_until_close,
    take_profit_threshold_perc: backtest.take_profit_threshold_perc,
    stop_loss_threshold_perc: backtest.stop_loss_threshold_perc,
    use_time_based_close: backtest.use_time_based_close,
    use_profit_based_close: backtest.use_profit_based_close,
    use_stop_loss_based_close: backtest.use_stop_loss_based_close,
  };
};

export const DeployStrategyForm = (props: Props) => {
  const { deployStrategyDrawer } = props;
  const { backtestId, datasetName } = usePathParams<PathParams>();
  const { getPredServAPIKey } = useAppContext();
  const backtestQuery = useBacktestById(Number(backtestId));
  const datasetQuery = useDatasetQuery(datasetName);

  const toast = useToast();

  if (!backtestQuery || !backtestQuery.data || !backtestQuery.data.data) {
    return null;
  }

  const backtest = backtestQuery.data.data;

  const onSubmit = async (form: DeployStratForm) => {
    if (!datasetQuery.data) {
      return null;
    }

    const res = await deployStrategyReq(getPredServAPIKey(), {
      ...form,
      data_transformations: datasetQuery.data.data_transformations,
    });

    if (res.status === 200) {
      toast({
        title: `Deployed strategy ${form.name}`,
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      backtestDiskManager.save(form);
      deployStrategyDrawer.onClose();
    } else {
      toast({
        title: `Deploying strategy failed. Error code: ${res.status}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };

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
            {() => {
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
                        {({ form, field }) => {
                          return (
                            <WithLabel>
                              <ChakraInput
                                label="Strategy name"
                                onChange={(value: string) =>
                                  form.setFieldValue(formKeys.name, value)
                                }
                                value={field.value}
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.symbol}>
                        {({ form, field }) => {
                          return (
                            <WithLabel>
                              <ChakraInput
                                label="Symbol"
                                onChange={(value: string) =>
                                  form.setFieldValue(formKeys.symbol, value)
                                }
                                value={field.value}
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.baseAsset}>
                        {({ form, field }) => {
                          return (
                            <WithLabel>
                              <ChakraInput
                                label="Base asset"
                                onChange={(value: string) =>
                                  form.setFieldValue(formKeys.baseAsset, value)
                                }
                                value={field.value}
                              />
                            </WithLabel>
                          );
                        }}
                      </Field>
                    </div>
                    <div>
                      <Field name={formKeys.quoteAsset}>
                        {({ form, field }) => {
                          return (
                            <WithLabel>
                              <ChakraInput
                                label="Quote asset"
                                onChange={(value: string) =>
                                  form.setFieldValue(formKeys.quoteAsset, value)
                                }
                                value={field.value}
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
                      onClick={deployStrategyDrawer.onClose}
                    >
                      Cancel
                    </Button>
                    <Button type="submit" marginTop={"16px"}>
                      Deploy
                    </Button>
                  </div>
                </Form>
              );
            }}
          </Formik>
        </div>
      </ChakraDrawer>
    </div>
  );
};
