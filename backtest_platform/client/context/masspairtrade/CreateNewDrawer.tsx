import React, { useRef, useState } from "react";
import { useMassBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { Field, Form, Formik, FormikProps } from "formik";
import {
  BACKTEST_FORM_LABELS,
  getBacktestFormDefaultKeys,
  getBacktestFormDefaults,
} from "../../utils/backtest";
import { DISK_KEYS, DiskManager } from "../../utils/disk";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";
import { CodeEditor } from "../../components/CodeEditor";
import { CODE_PRESET_CATEGORY, GET_KLINE_OPTIONS } from "../../utils/constants";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../components/SelectFilter";
import { binanceTickSelectOptions } from "../../pages/Datasets";
import {
  useBinanceTickersQuery,
  useDataTransformations,
} from "../../clients/queries/queries";
import { MultiValue } from "react-select";
import {
  Button,
  FormControl,
  FormLabel,
  NumberInput,
  NumberInputField,
  Select,
  Spinner,
  Switch,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import { ValidationSplitSlider } from "../../components/ValidationSplitSlider";
import { ChakraPopover } from "../../components/chakra/popover";
import { BUTTON_VARIANTS } from "../../theme";
import { SelectBulkSimPairsBody } from "../backtest/run-on-many-pairs";
import { DataTransformationControls } from "../../components/DataTransformationsControls";
import { createMassPairTradeSim } from "../../clients/requests";
import {
  PAIR_TRADE_BUY_DEFAULT,
  PAIR_TRADE_EXIT_DEFAULT,
  PAIR_TRADE_SELL_DEFAULT,
} from "../../utils/code";
import { ChakraNumberStepper } from "../../components/ChakraNumberStepper";

const backtestDiskManager = new DiskManager(DISK_KEYS.mass_long_short_form);

export interface BodyMassPairTradeSim {
  datasets: string[];
  name: string;
  use_time_based_close: boolean;
  use_profit_based_close: boolean;
  use_stop_loss_based_close: boolean;
  trading_fees_perc: number;
  slippage_perc: number;
  short_fee_hourly: number;
  take_profit_threshold_perc: number;
  stop_loss_threshold_perc: number;
  klinesUntilClose: number;
  data_transformations: number[];
  sell_cond: string;
  buy_cond: string;
  exit_cond: string;
  max_simultaneous_positions: number;
  max_leverage_ratio: number;
  candle_interval: string;
  fetch_latest_data: boolean;
}

export interface BacktestFormDefaults {
  backtestName: string;
  useTimeBasedClose: boolean;
  useProfitBasedClose: boolean;
  useStopLossBasedClose: boolean;
  klinesUntilClose: number;
  tradingFees: number;
  slippage: number;
  shortFeeHourly: number;
  takeProfitThresholdPerc: number;
  stopLossThresholdPerc: number;
  backtestDataRange: [number, number];
}

interface FormValues extends BacktestFormDefaults {
  buy_criteria: string;
  short_criteria: string;
  exit_cond: string;
  pairs: MultiValue<OptionType>;
  max_simultaneous_positions: number;
  max_leverage_ratio: number;
  data_transformations: number[];
  use_latest_data: boolean;
  candle_interval: string;
}

const getFormInitialValues = () => {
  const prevForm = backtestDiskManager.read();
  if (prevForm === null) {
    return {
      buy_criteria: PAIR_TRADE_BUY_DEFAULT(),
      short_criteria: PAIR_TRADE_SELL_DEFAULT(),
      exit_cond: PAIR_TRADE_EXIT_DEFAULT(),
      pairs: [],
      max_simultaneous_positions: 15,
      max_leverage_ratio: 2.5,
      data_transformations: [],
      use_latest_data: false,
      candle_interval: "",
      ...getBacktestFormDefaults(),
    };
  }
  return {
    ...prevForm,
    backtestName: "",
  };
};

const formKeys = {
  buy_criteria: "buy_criteria",
  sell_criteria: "short_criteria",
  exit_cond: "exit_cond",
  pairs: "pairs",
  max_simultaneous_positions: "max_simultaneous_positions",
  max_leverage_ratio: "max_leverage_ratio",
  data_transformations: "data_transformations",
  useLatestData: "use_latest_data",
  candleInterval: "candle_interval",
  ...getBacktestFormDefaultKeys(),
};

export const BulkLongShortCreateNew = () => {
  const { createNewDrawer } = useMassBacktestContext();
  const binanceTickersQuery = useBinanceTickersQuery();
  const formikRef = useRef<FormikProps<any>>(null);
  const presetsPopover = useDisclosure();
  const toast = useToast();
  const dataTransformationsQuery = useDataTransformations();

  if (!binanceTickersQuery.data) {
    return (
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...createNewDrawer}
      >
        <Spinner />
      </ChakraDrawer>
    );
  }

  const onSubmit = async (values: FormValues) => {
    values.data_transformations = values.data_transformations.filter((a) => {
      let found = false;
      dataTransformationsQuery.data?.forEach((b) => {
        if (a == b.id) {
          found = true;
        }
      });
      return found;
    });
    const body = {
      datasets: values.pairs.map((item) => item.value),
      name: values.backtestName,
      use_time_based_close: values.useTimeBasedClose,
      use_profit_based_close: values.useProfitBasedClose,
      use_stop_loss_based_close: values.useStopLossBasedClose,
      trading_fees_perc: values.tradingFees,
      slippage_perc: values.slippage,
      short_fee_hourly: values.shortFeeHourly,
      take_profit_threshold_perc: values.takeProfitThresholdPerc,
      stop_loss_threshold: values.stopLossThresholdPerc,
      klinesUntilClose: values.klinesUntilClose,
      data_transformations: values.data_transformations,
      sell_cond: values.short_criteria,
      buy_cond: values.buy_criteria,
      exit_cond: values.exit_cond,
      max_simultaneous_positions: values.max_simultaneous_positions,
      max_leverage_ratio: values.max_leverage_ratio,
      candle_interval: values.candle_interval,
      stop_loss_threshold_perc: values.stopLossThresholdPerc,
      fetch_latest_data: values.use_latest_data,
      backtest_data_range: values.backtestDataRange,
      klines_until_close: values.klinesUntilClose,
    };

    const res = await createMassPairTradeSim(body);
    if (res.status === 200) {
      toast({
        title: "Started mass pair trade simulation",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      backtestDiskManager.save(values);
    }
  };

  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...createNewDrawer}
      >
        <div>
          <Formik
            onSubmit={(values) => {
              onSubmit(values);
            }}
            initialValues={getFormInitialValues()}
            innerRef={formikRef}
            enableReinitialize
          >
            {({ values, setFieldValue }) => (
              <Form>
                <div
                  style={{
                    display: "flex",
                    alignItems: "end",
                    gap: "16px",
                  }}
                >
                  <div>
                    <Field name={formKeys.backtestName}>
                      {({ form }) => {
                        return (
                          <WithLabel>
                            <ChakraInput
                              label={BACKTEST_FORM_LABELS.name}
                              onChange={(value: string) =>
                                form.setFieldValue(formKeys.backtestName, value)
                              }
                            />
                          </WithLabel>
                        );
                      }}
                    </Field>
                  </div>
                  <div style={{ marginTop: "16px", width: "600px" }}>
                    <WithLabel
                      label={
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                          }}
                        >
                          <div>Select pairs</div>

                          <div>
                            <ChakraPopover
                              {...presetsPopover}
                              setOpen={presetsPopover.onOpen}
                              body={
                                <SelectBulkSimPairsBody
                                  onSelect={(values) =>
                                    setFieldValue(formKeys.pairs, values)
                                  }
                                />
                              }
                              headerText="Select pairs from a preset"
                            >
                              <Button variant={BUTTON_VARIANTS.nofill}>
                                Presets
                              </Button>
                            </ChakraPopover>
                          </div>
                        </div>
                      }
                    >
                      <Field
                        name={formKeys.pairs}
                        as={SelectWithTextFilter}
                        options={binanceTickSelectOptions(
                          binanceTickersQuery.data.res.pairs
                        )}
                        onChange={(selectedOptions: MultiValue<OptionType>) =>
                          setFieldValue(formKeys.pairs, selectedOptions)
                        }
                        isMulti={true}
                        closeMenuOnSelect={false}
                        value={values.pairs}
                      />
                    </WithLabel>
                  </div>
                </div>

                <div
                  style={{ marginTop: "16px", display: "flex", gap: "16px" }}
                >
                  <Field name={formKeys.useLatestData}>
                    {({ field, form }) => {
                      return (
                        <WithLabel label={"Download latest data"}>
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue(
                                formKeys.useLatestData,
                                !field.value
                              )
                            }
                          />
                        </WithLabel>
                      );
                    }}
                  </Field>
                </div>

                <div style={{ marginTop: "16px", maxWidth: "250px" }}>
                  <FormControl>
                    <FormLabel fontSize={"x-large"}>Candle interval</FormLabel>
                    <Field name={formKeys.candleInterval} as={Select}>
                      {GET_KLINE_OPTIONS().map((item) => (
                        <option key={item} value={item}>
                          {item}
                        </option>
                      ))}
                    </Field>
                  </FormControl>
                </div>

                <Field name={formKeys.data_transformations}>
                  {({ field, form }) => {
                    return (
                      <div style={{ marginTop: "16px" }}>
                        <DataTransformationControls
                          selectedTransformations={field.value}
                          onSelect={(newState) => {
                            form.setFieldValue(
                              formKeys.data_transformations,
                              newState
                            );
                          }}
                        />
                      </div>
                    );
                  }}
                </Field>

                <div>
                  <Field name={formKeys.buy_criteria}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) => {
                            form.setFieldValue(formKeys.buy_criteria, newState);
                          }}
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={BACKTEST_FORM_LABELS.long_short_buy_condition}
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
                  <Field name={formKeys.sell_criteria}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(formKeys.sell_criteria, newState)
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={BACKTEST_FORM_LABELS.long_short_sell_condition}
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
                <div>
                  <Field name={formKeys.exit_cond}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(formKeys.exit_cond, newState)
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={"Pair exit condition"}
                          codeContainerStyles={{ width: "100%" }}
                          height={"250px"}
                          presetCategory={
                            CODE_PRESET_CATEGORY.backtest_pair_trade_exit_cond
                          }
                        />
                      );
                    }}
                  </Field>
                </div>

                <div>
                  <Field name={formKeys.max_simultaneous_positions}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Max simultaneous positions"}
                          containerStyles={{
                            maxWidth: "300px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={1}
                            min={0}
                            value={field.value}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                formKeys.max_simultaneous_positions,
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
                </div>

                <div>
                  <Field name={formKeys.max_leverage_ratio}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Max leverage ratio"}
                          containerStyles={{
                            maxWidth: "200px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={0.1}
                            max={3.0}
                            min={0.0}
                            precision={2}
                            value={field.value}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                formKeys.max_leverage_ratio,
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
                <div style={{ marginTop: "16px" }}>
                  <Field name={formKeys.useTimeBasedClose}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.use_time_based_close}
                        >
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
                {values.useTimeBasedClose && (
                  <Field name={formKeys.klinesUntilClose}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.klines_until_close}
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
                                formKeys.klinesUntilClose,
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
                  <Field name={formKeys.useProfitBasedClose}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.use_profit_based_close}
                        >
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
                {values.useProfitBasedClose && (
                  <Field name={formKeys.takeProfitThresholdPerc}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.take_profit_threshold}
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
                                formKeys.takeProfitThresholdPerc,
                                parseFloat(valueString)
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
                  <Field name={formKeys.useStopLossBasedClose}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.use_stop_loss_based_close}
                        >
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
                {values.useStopLossBasedClose && (
                  <Field name={formKeys.stopLossThresholdPerc}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.stop_loss_threshold}
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
                                formKeys.stopLossThresholdPerc,
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
                  <Field name={formKeys.tradingFees}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.trading_fees}
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
                                formKeys.tradingFees,
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

                  <Field name={formKeys.slippage}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.slippage}
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
                                formKeys.slippage,
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
                  <Field name={formKeys.shortFeeHourly}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.shorting_fees_hourly}
                          containerStyles={{
                            maxWidth: "200px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={0.0000005}
                            min={0}
                            value={field.value}
                            precision={6}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                formKeys.shortFeeHourly,
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
                <div style={{ marginTop: "16px" }}>
                  <div style={{ width: "400px" }}>
                    <Field name={formKeys.backtestDataRange}>
                      {({ field, form }) => {
                        return (
                          <ValidationSplitSlider
                            sliderValue={field.value}
                            formLabelText={
                              BACKTEST_FORM_LABELS.backtest_data_range
                            }
                            setSliderValue={(newVal: number[]) =>
                              form.setFieldValue(
                                formKeys.backtestDataRange,
                                newVal
                              )
                            }
                          />
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
