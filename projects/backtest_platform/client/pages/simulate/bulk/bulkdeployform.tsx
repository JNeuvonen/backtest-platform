import React from "react";
import { DISK_KEYS, DiskManager } from "../../../utils/disk";
import {
  BacktestObject,
  DataTransformation,
  Dataset,
} from "../../../clients/queries/response-types";
import {
  getIntervalLengthInMs,
  getTradeQuantityPrecision,
  inferAssets,
} from "../../../utils/binance";
import { FETCH_DATASOURCES_DEFAULT } from "../../../utils/code";
import { useAppContext } from "../../../context/app";
import {
  Button,
  NumberInput,
  NumberInputField,
  Switch,
  UseDisclosureReturn,
  useToast,
} from "@chakra-ui/react";
import { ChakraDrawer } from "../../../components/chakra/Drawer";
import { Field, Form, Formik } from "formik";
import { WithLabel } from "../../../components/form/WithLabel";
import { ChakraInput } from "../../../components/chakra/input";
import { CodeEditor } from "../../../components/CodeEditor";
import { CODE_PRESET_CATEGORY } from "../../../utils/constants";
import { ChakraNumberStepper } from "../../../components/ChakraNumberStepper";
import { BUTTON_VARIANTS } from "../../../theme";
import { useDatasetQuery } from "../../../clients/queries/queries";
import {
  deployStrategyGroup,
  deployStrategyReq,
  fetchDataset,
} from "../../../clients/requests";
import { SymbolDeployInfo } from "common_js";

const backtestDiskManager = new DiskManager(
  DISK_KEYS.bulk_deploy_strategy_form
);

const formKeys = {
  name: "name",
  enterTradeCode: "enter_trade_code",
  exitTradeCode: "exit_trade_code",
  fetchDatasourcesCode: "fetch_datasources_code",
  priority: "priority",
  klineSizeMs: "kline_size_ms",
  maximumKlinesHoldTime: "maximum_klines_hold_time",
  numReqKlines: "num_req_klines",
  allocatedSizePerc: "allocated_size_perc",
  takeProfitThresholdPerc: "take_profit_threshold_perc",
  stopLossThresholdPerc: "stop_loss_threshold_perc",
  minimumTimeBetweenTradesMs: "minimum_time_between_trades_ms",
  shouldCalcStopsOnPredServ: "should_calc_stops_on_pred_serv",
  useTimeBasedClose: "use_time_based_close",
  useProfitBasedClose: "use_profit_based_close",
  useStopLossBasedClose: "use_stop_loss_based_close",
  useTakerOrder: "use_taker_order",
  isLeverageAllowed: "is_leverage_allowed",
  isShortSellingStrategy: "is_short_selling_strategy",
  isPaperTradeMode: "is_paper_trade_mode",
};

export interface BulkDeployDirectionalStratForm {
  name?: string;
  enter_trade_code: string;
  exit_trade_code: string;
  candle_interval: string;
  fetch_datasources_code: string;
  priority: number;
  kline_size_ms: number;
  maximum_klines_hold_time: number;
  num_req_klines: number;
  allocated_size_perc: number;
  take_profit_threshold_perc: number;
  stop_loss_threshold_perc: number;
  should_calc_stops_on_pred_serv: boolean;
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

const getFormInitialValues = (
  backtest: BacktestObject
): BulkDeployDirectionalStratForm => {
  const prevForm = backtestDiskManager.read();
  if (prevForm === null) {
    return {
      name: "",
      enter_trade_code: backtest.open_trade_cond,
      exit_trade_code: backtest.close_trade_cond,
      fetch_datasources_code: FETCH_DATASOURCES_DEFAULT,
      candle_interval: backtest.candle_interval,
      priority: 1,
      kline_size_ms: getIntervalLengthInMs(backtest.candle_interval),
      maximum_klines_hold_time: backtest.klines_until_close,
      num_req_klines: 1000,
      allocated_size_perc: 25,
      take_profit_threshold_perc: backtest.take_profit_threshold_perc,
      stop_loss_threshold_perc: backtest.stop_loss_threshold_perc,
      should_calc_stops_on_pred_serv: false,
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
    kline_size_ms: getIntervalLengthInMs(backtest.candle_interval),
    candle_interval: backtest.candle_interval,
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

interface Props {
  bulkDeployDrawer: UseDisclosureReturn;
  originalBacktest: BacktestObject;
  backtests: BacktestObject[];
}

export const BulkDeployDirectionalStratForm = (props: Props) => {
  const { bulkDeployDrawer, originalBacktest, backtests } = props;
  const { getPredServAPIKey } = useAppContext();
  const datasetQuery = useDatasetQuery(originalBacktest.dataset_name);
  const toast = useToast();

  if (!datasetQuery.data) {
    return null;
  }

  const getSymbolInfoList = async () => {
    const ret: SymbolDeployInfo[] = [];

    const dataset = await fetchDataset(originalBacktest.dataset_name);

    if (dataset) {
      const precision = await getTradeQuantityPrecision(dataset.symbol);
      const { quoteAsset, baseAsset } = inferAssets(dataset.symbol);
      ret.push({
        symbol: dataset.symbol,
        base_asset: baseAsset,
        quote_asset: quoteAsset,
        trade_quantity_precision: precision,
      });
    }

    for (const item of backtests) {
      const dataset = await fetchDataset(item.dataset_name);

      if (!dataset) {
        continue;
      }
      const precision = await getTradeQuantityPrecision(dataset.symbol);
      const { quoteAsset, baseAsset } = inferAssets(dataset.symbol);
      ret.push({
        symbol: dataset.symbol,
        base_asset: baseAsset,
        quote_asset: quoteAsset,
        trade_quantity_precision: precision,
      });
    }
    return ret;
  };

  const onSubmit = async (form: BulkDeployDirectionalStratForm) => {
    if (!form.name) {
      toast({
        title: "Name should not be empty",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    if (datasetQuery.data === null) {
      return null;
    }

    const dataTransformations = datasetQuery.data
      ? datasetQuery.data.data_transformations
      : [];

    if (dataTransformations.length === 0) {
      return;
    }

    const symbolInfo = await getSymbolInfoList();

    await deployStrategyGroup(getPredServAPIKey(), {
      ...form,
      symbols: symbolInfo,
      strategy_group: form.name,
    });

    backtestDiskManager.save(form);
    bulkDeployDrawer.onClose();
    toast({
      title: "All strategies deployed successfully",
      status: "success",
      duration: 5000,
      isClosable: true,
    });
  };

  return (
    <ChakraDrawer
      {...bulkDeployDrawer}
      title={"Bulk deploy strategy"}
      drawerContentStyles={{ maxWidth: "80%" }}
    >
      <div>
        <Formik
          initialValues={getFormInitialValues(originalBacktest)}
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
                              containerStyle={{ width: "500px" }}
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
                          readOnly={true}
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
                          readOnly={true}
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(formKeys.exitTradeCode, newState)
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

                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "16px",
                    marginTop: "16px",
                  }}
                >
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
                              <ChakraNumberStepper />
                            </NumberInput>
                          </WithLabel>
                        );
                      }}
                    </Field>
                  </div>
                  <div>
                    <Field name={formKeys.numReqKlines}>
                      {({ field, form }) => {
                        return (
                          <WithLabel
                            label={"Num of required candles"}
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
                                  formKeys.numReqKlines,
                                  parseInt(valueString)
                                )
                              }
                            >
                              <NumberInputField />
                              <ChakraNumberStepper />
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
                              isDisabled={true}
                              onChange={(valueString) =>
                                form.setFieldValue(
                                  formKeys.maximumKlinesHoldTime,
                                  parseInt(valueString)
                                )
                              }
                            >
                              <NumberInputField />
                              <ChakraNumberStepper />
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
                            label={"Size (%) per symbol"}
                            containerStyles={{
                              maxWidth: "200px",
                            }}
                          >
                            <NumberInput
                              step={0.1}
                              min={0}
                              value={field.value}
                              onChange={(valueString) =>
                                form.setFieldValue(
                                  formKeys.allocatedSizePerc,
                                  parseFloat(valueString)
                                )
                              }
                            >
                              <NumberInputField />
                              <ChakraNumberStepper />
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
                              <ChakraNumberStepper />
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
                              isDisabled={true}
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
                    <Field name={formKeys.shouldCalcStopsOnPredServ}>
                      {({ field, form }) => {
                        return (
                          <WithLabel label={"Calculate stops on new kline"}>
                            <Switch
                              isChecked={field.value}
                              onChange={() =>
                                form.setFieldValue(
                                  formKeys.shouldCalcStopsOnPredServ,
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
                              isDisabled={true}
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
                              isDisabled={true}
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
                              isDisabled={true}
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
                    onClick={bulkDeployDrawer.onClose}
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
  );
};
