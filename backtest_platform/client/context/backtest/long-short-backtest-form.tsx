import React, { useRef, useState } from "react";
import { useBacktestContext } from ".";
import { useDatasetQuery } from "../../clients/queries/queries";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { usePathParams } from "../../hooks/usePathParams";
import {
  Button,
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
import { ShowColumnModal } from "./show-columns-modal";
import { BacktestFormControls } from "./backtest-form-controls";
import { Field, Form, Formik, FormikProps } from "formik";
import { DISK_KEYS, DiskManager } from "../../utils/disk";
import { useForceUpdate } from "../../hooks/useForceUpdate";
import { ChakraModal } from "../../components/chakra/modal";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";
import {
  CREATE_COLUMNS_DEFAULT,
  LONG_SHORT_BUY_COND_DEFAULT,
  LONG_SHORT_SELL_COND_DEFAULT,
} from "../../utils/code";
import { execPythonOnDataset } from "../../clients/requests";
import { CodeEditor } from "../../components/CodeEditor";
import { CODE_PRESET_CATEGORY } from "../../utils/constants";
import {
  BACKTEST_FORM_LABELS,
  getBacktestFormDefaultKeys,
  getBacktestFormDefaults,
} from "../../utils/backtest";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";
import { ValidationSplitSlider } from "../../components/ValidationSplitSlider";
import { BUTTON_VARIANTS } from "../../theme";

type PathParams = {
  datasetName: string;
};

const backtestDiskManager = new DiskManager(DISK_KEYS.backtest_form);

const formKeys = {
  buy_criteria: "buy_criteria",
  sell_criteria: "short_criteria",
  ...getBacktestFormDefaultKeys(),
};

const getFormInitialValues = () => {
  return {
    name: "",
    buy_criteria: LONG_SHORT_BUY_COND_DEFAULT(),
    short_criteria: LONG_SHORT_SELL_COND_DEFAULT(),
    ...getBacktestFormDefaults(),
  };
};

export const LongShortBacktestForm = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { longShortFormDrawer } = useBacktestContext();
  const columnsModal = useDisclosure();
  const runPythonModal = useDisclosure();
  const { data: datasetQuery, refetch: refetchDataset } =
    useDatasetQuery(datasetName);
  const formikRef = useRef<FormikProps<any>>(null);
  const [createColumnsCode, setCreateColumnsCode] = useState(
    CREATE_COLUMNS_DEFAULT()
  );
  const forceUpdate = useForceUpdate();
  const toast = useToast();

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

  if (!datasetQuery) return <Spinner />;

  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...longShortFormDrawer}
      >
        <div>
          <ShowColumnModal
            columns={datasetQuery.columns}
            datasetName={datasetName}
            columnsModal={columnsModal}
          />

          <BacktestFormControls
            columnsModal={columnsModal}
            formikRef={formikRef}
            backtestDiskManager={backtestDiskManager}
            forceUpdate={forceUpdate}
            runPythonModal={runPythonModal}
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
              codeContainerStyles={{ width: "100%" }}
              height={"250px"}
              presetCategory={CODE_PRESET_CATEGORY.backtest_create_columns}
            />
          </ChakraModal>

          <Formik
            onSubmit={(values) => {}}
            initialValues={getFormInitialValues()}
            innerRef={formikRef}
            enableReinitialize
          >
            {({ values }) => (
              <Form>
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
                            <NumberInputStepper>
                              <NumberIncrementStepper />
                              <NumberDecrementStepper />
                            </NumberInputStepper>
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
                            <NumberInputStepper>
                              <NumberIncrementStepper color={"white"} />
                              <NumberDecrementStepper color={"white"} />
                            </NumberInputStepper>
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
                            <NumberInputStepper>
                              <NumberIncrementStepper />
                              <NumberDecrementStepper />
                            </NumberInputStepper>
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
                    onClick={longShortFormDrawer.onClose}
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
