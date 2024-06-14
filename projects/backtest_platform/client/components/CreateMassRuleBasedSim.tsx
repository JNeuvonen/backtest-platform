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

interface Props {
  drawerControls: UseDisclosureReturn;
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
  tradingFees: "tradingFees",
  shortFeeHourly: "shortFeeHourly",
  slippage: "slippage",
};

const getFormInitialValues = () => {
  return {
    [formKeys.backtestName]: "",
    [formKeys.useLatestData]: false,
    [formKeys.startDate]: null,
    [formKeys.endDate]: null,
    [formKeys.dataTransformations]: [],
    [formKeys.enterCriteria]: "",
    [formKeys.exitCriteria]: "",
    [formKeys.isShortSellingStrategy]: false,
    [formKeys.useTimeBasedClose]: false,
    [formKeys.useProfitBasedClose]: false,
    [formKeys.useStopLossBasedClose]: false,
    [formKeys.klinesUntilClose]: 1,
    [formKeys.takeProfitThresholdPerc]: 1,
    [formKeys.stopLossThresholdPerc]: 1,
    [formKeys.tradingFees]: 0.1,
    [formKeys.slippage]: 0,
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

export const CreateMassRuleBasedSim = ({ drawerControls }: Props) => {
  const selectTransformationsModal = useDisclosure();

  return (
    <div>
      <ChakraDrawer
        title={"Create a new backtest"}
        drawerContentStyles={{ maxWidth: "80%" }}
        {...drawerControls}
      >
        <Formik
          onSubmit={(values) => {}}
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
                          form.setFieldValue(formKeys.enterCriteria, newState);
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

                {values.useShorts && (
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
            </Form>
          )}
        </Formik>
      </ChakraDrawer>
    </div>
  );
};
