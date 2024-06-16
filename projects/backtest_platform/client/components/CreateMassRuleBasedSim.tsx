import {
  NumberInput,
  NumberInputField,
  Switch,
  UseDisclosureReturn,
  useDisclosure,
} from "@chakra-ui/react";
import React from "react";
import { ChakraDrawer } from "./chakra/Drawer";
import { Field, Form, Formik } from "formik";
import { WithLabel } from "./form/WithLabel";
import { ChakraInput } from "./chakra/input";
import { BACKTEST_FORM_LABELS } from "../utils/backtest";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { DataTransformationControls } from "./DataTransformationsControls";
import { CodeEditor } from "./CodeEditor";
import { ChakraNumberStepper } from "./ChakraNumberStepper";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { ENTER_TRADE_DEFAULT, EXIT_LONG_TRADE_DEFAULT } from "../utils/code";
import { DiskManager } from "common_js";
import { DISK_KEYS } from "../utils/disk";

const backtestDiskManager = new DiskManager(DISK_KEYS.rule_based_mass_backtest);

interface Props {
  drawerControls: UseDisclosureReturn;
  onSubmit: (formValues: any) => void;
}

export interface BacktestOnUniverseFormValues {
  backtestName: string;
  useLatestData: boolean;
  startDate: Date | null;
  endDate: Date | null;
  dataTransformations: number[];
  enterCriteria: string;
  exitCriteria: string;
  isShortSellingStrategy: boolean;
  useTimeBasedClose: boolean;
  useProfitBasedClose: boolean;
  useStopLossBasedClose: boolean;
  klinesUntilClose: number;
  takeProfitThresholdPerc: number;
  stopLossThresholdPerc: number;
  tradingFees: number;
  slippage: number;
  allocationPerSymbol: number;
  shortFeeHourly: number;
}

const formKeys = {
  backtestName: "backtestName",
  useLatestData: "useLatestData",
  enterCriteria: "enterCriteria",
  exitCriteria: "exitCriteria",
  startDate: "startDate",
  endDate: "endDate",
  dataTransformations: "dataTransformations",
  isShortSellingStrategy: "isShortSellingStrategy",
  useTimeBasedClose: "useTimeBasedClose",
  useProfitBasedClose: "useProfitBasedClose",
  useStopLossBasedClose: "useStopLossBasedClose",
  klinesUntilClose: "klinesUntilClose",
  takeProfitThresholdPerc: "takeProfitThresholdPerc",
  stopLossThresholdPerc: "stopLossThresholdPerc",
  allocationPerSymbol: "allocationPerSymbol",
  tradingFees: "tradingFees",
  shortFeeHourly: "shortFeeHourly",
  slippage: "slippage",
};

const getFormInitialValues = () => {
  const prevForm = backtestDiskManager.read();

  if (prevForm === null) {
    return {
      [formKeys.backtestName]: "",
      [formKeys.useLatestData]: false,
      [formKeys.startDate]: null,
      [formKeys.endDate]: null,
      [formKeys.dataTransformations]: [],
      [formKeys.enterCriteria]: ENTER_TRADE_DEFAULT(),
      [formKeys.exitCriteria]: EXIT_LONG_TRADE_DEFAULT(),
      [formKeys.isShortSellingStrategy]: false,
      [formKeys.useTimeBasedClose]: false,
      [formKeys.useProfitBasedClose]: false,
      [formKeys.useStopLossBasedClose]: false,
      [formKeys.klinesUntilClose]: 1,
      [formKeys.takeProfitThresholdPerc]: 1,
      [formKeys.stopLossThresholdPerc]: 1,
      [formKeys.tradingFees]: 0.1,
      [formKeys.slippage]: 0,
      [formKeys.allocationPerSymbol]: 0.5,
      [formKeys.shortFeeHourly]: 0.00016,
    };
  }

  return {
    ...prevForm,
    [formKeys.startDate]: prevForm[formKeys.startDate]
      ? new Date(prevForm[formKeys.startDate])
      : null,
    [formKeys.endDate]: prevForm[formKeys.endDate]
      ? new Date(prevForm[formKeys.endDate])
      : null,
  };
};

const FormikDatePicker = ({ field, form, ...props }) => {
  return (
    <DatePicker
      {...field}
      {...props}
      selected={(field.value && new Date(field.value)) || null}
      onChange={(date) => form.setFieldValue(field.name, date)}
    />
  );
};

export const CreateMassRuleBasedSim = ({ drawerControls, onSubmit }: Props) => {
  return (
    <div>
      <ChakraDrawer
        title={"Create a new backtest"}
        drawerContentStyles={{ maxWidth: "80%" }}
        {...drawerControls}
      >
        <Formik
          onSubmit={(values) => {
            backtestDiskManager.save(values);
            onSubmit(values);
          }}
          initialValues={getFormInitialValues()}
        >
          {({ values }) => (
            <Form>
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
              <div style={{ marginTop: "16px", display: "flex", gap: "16px" }}>
                <div>
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
                <div>
                  <WithLabel label={"Start date"}>
                    <Field
                      name={formKeys.startDate}
                      component={FormikDatePicker}
                    />
                  </WithLabel>
                </div>
                <div>
                  <WithLabel label={"End date"}>
                    <Field
                      name={formKeys.endDate}
                      component={FormikDatePicker}
                    />
                  </WithLabel>
                </div>
              </div>

              <div>
                <Field name={formKeys.dataTransformations}>
                  {({ field, form }) => {
                    return (
                      <div style={{ marginTop: "16px" }}>
                        <DataTransformationControls
                          selectedTransformations={field.value}
                          onSelect={(newState) => {
                            form.setFieldValue(
                              formKeys.dataTransformations,
                              newState
                            );
                          }}
                        />
                      </div>
                    );
                  }}
                </Field>
              </div>

              <div>
                <Field name={formKeys.enterCriteria}>
                  {({ field, form }) => {
                    return (
                      <CodeEditor
                        code={field.value}
                        setCode={(newState) => {
                          form.setFieldValue(formKeys.enterCriteria, newState);
                        }}
                        style={{ marginTop: "16px" }}
                        fontSize={13}
                        label={"Enter criteria"}
                        codeContainerStyles={{ width: "100%" }}
                        height={"250px"}
                      />
                    );
                  }}
                </Field>
              </div>
              <div>
                <Field name={formKeys.exitCriteria}>
                  {({ field, form }) => {
                    return (
                      <CodeEditor
                        code={field.value}
                        setCode={(newState) => {
                          form.setFieldValue(formKeys.exitCriteria, newState);
                        }}
                        style={{ marginTop: "16px" }}
                        fontSize={13}
                        label={"Exit criteria"}
                        codeContainerStyles={{ width: "100%" }}
                        height={"250px"}
                      />
                    );
                  }}
                </Field>
              </div>
              <div style={{ marginTop: "16px" }}>
                <Field name={formKeys.isShortSellingStrategy}>
                  {({ field, form }) => {
                    return (
                      <WithLabel
                        label={BACKTEST_FORM_LABELS.is_short_selling_strategy}
                      >
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

              <div style={{ marginTop: "16px" }}>
                <Field name={formKeys.allocationPerSymbol}>
                  {({ field, form }) => {
                    return (
                      <WithLabel
                        label={"Allocation % per symbol"}
                        containerStyles={{
                          maxWidth: "200px",
                          marginTop: "16px",
                        }}
                      >
                        <NumberInput
                          step={0.25}
                          min={0}
                          value={field.value}
                          precision={3}
                          onChange={(valueString) =>
                            form.setFieldValue(
                              formKeys.allocationPerSymbol,
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
                          step={0.1}
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
                          <ChakraNumberStepper />
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
                          min={0}
                          step={0.1}
                          value={field.value}
                          onChange={(valueString) =>
                            form.setFieldValue(
                              formKeys.stopLossThresholdPerc,
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

                {values[formKeys.isShortSellingStrategy] && (
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
                )}
              </div>

              <FormSubmitBar />
            </Form>
          )}
        </Formik>
      </ChakraDrawer>
    </div>
  );
};
