import React, { useRef, useState } from "react";
import { useBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";
import {
  CREATE_COLUMNS_DEFAULT,
  ENTER_TRADE_DEFAULT,
  EXIT_LONG_TRADE_DEFAULT,
} from "../../utils/code";
import {
  createManualBacktest,
  execPythonOnDataset,
  setBacktestPriceColumn,
} from "../../clients/requests";
import { usePathParams } from "../../hooks/usePathParams";
import { useDatasetQuery } from "../../clients/queries/queries";
import {
  Button,
  NumberInput,
  NumberInputField,
  Spinner,
  Switch,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import { ChakraModal } from "../../components/chakra/modal";
import { ChakraPopover } from "../../components/chakra/popover";
import { SelectColumnPopover } from "../../components/SelectTargetColumnPopover";
import { getDatasetColumnOptions } from "../../utils/dataset";
import { CodeEditor } from "../../components/CodeEditor";
import { CODE_PRESET_CATEGORY, COPILOT_LINKS } from "../../utils/constants";
import { BUTTON_VARIANTS } from "../../theme";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";
import { Field, Formik, Form, FormikProps } from "formik";
import { ValidationSplitSlider } from "../../components/ValidationSplitSlider";
import { DISK_KEYS, DiskManager } from "../../utils/disk";
import { BacktestFormControls } from "./backtest-form-controls";
import { ShowColumnModal } from "./show-columns-modal";
import {
  getBacktestFormDefaults,
  getBacktestFormDefaultKeys,
  BACKTEST_FORM_LABELS,
} from "../../utils/backtest";
import { ChakraNumberStepper } from "../../components/ChakraNumberStepper";

type PathParams = {
  datasetName: string;
};

const formKeys = {
  openTradeCode: "openTradeCode",
  closeTradeCode: "closeTradeCode",
  useShorts: "useShorts",
  ...getBacktestFormDefaultKeys(),
};

export interface BacktestFormValues {
  backtestName: string;
  closeTradeCode: string;
  openTradeCode: string;
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
  backtestDataRange: number[];
}

const backtestDiskManager = new DiskManager(DISK_KEYS.backtest_form);

const getFormInitialValues = () => {
  const prevForm = backtestDiskManager.read();
  if (prevForm === null) {
    return {
      openTradeCode: ENTER_TRADE_DEFAULT(),
      closeTradeCode: EXIT_LONG_TRADE_DEFAULT(),
      useShorts: false,
      ...getBacktestFormDefaults(),
    };
  }
  return {
    ...prevForm,
    backtestName: "",
  };
};

export const BacktestForm = () => {
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
  const formikRef = useRef<FormikProps<BacktestFormValues>>(null);

  const { data, refetch: refetchDataset } = useDatasetQuery(datasetName);
  const backtestPriceColumnPopover = useDisclosure();

  const submitNewBacktest = async (values: BacktestFormValues) => {
    if (!dataset) return;

    const res = await createManualBacktest({
      open_trade_cond: values.openTradeCode,
      close_trade_cond: values.closeTradeCode,
      is_short_selling_strategy: values.useShorts,
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
      backtest_data_range: values.backtestDataRange,
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
      backtestDiskManager.save(values);
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

  if (!dataset || !data) return <Spinner />;

  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...createNewDrawer}
      >
        <div>
          <ShowColumnModal
            datasetName={datasetName}
            columns={data.columns}
            columnsModal={columnsModal}
          />
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
                selectCallback={(newCol: string) => {
                  setBacktestPriceColumn(newCol, datasetName, () => {
                    toast({
                      title: "Changed price column",
                      status: "info",
                      duration: 5000,
                      isClosable: true,
                    });
                    backtestPriceColumnPopover.onClose();
                    refetchDataset();
                  });
                }}
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
              codeContainerStyles={{ width: "100%" }}
              height={"250px"}
              presetCategory={CODE_PRESET_CATEGORY.backtest_create_columns}
            />
          </ChakraModal>

          <BacktestFormControls
            columnsModal={columnsModal}
            formikRef={formikRef}
            backtestDiskManager={backtestDiskManager}
            forceUpdate={forceUpdate}
            runPythonModal={runPythonModal}
          />

          <Formik
            onSubmit={(values) => {
              submitNewBacktest(values);
            }}
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
                  <Field name={formKeys.openTradeCode}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(formKeys.openTradeCode, newState)
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={BACKTEST_FORM_LABELS.long_condition}
                          codeContainerStyles={{ width: "100%" }}
                          height={"250px"}
                          presetCategory={
                            CODE_PRESET_CATEGORY.backtest_long_cond
                          }
                          chatGptLink={COPILOT_LINKS.trade_criteria}
                        />
                      );
                    }}
                  </Field>
                </div>

                <div>
                  <Field name={formKeys.closeTradeCode}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(
                              formKeys.closeTradeCode,
                              newState
                            )
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={BACKTEST_FORM_LABELS.close_condition}
                          codeContainerStyles={{ width: "100%" }}
                          height={"250px"}
                          presetCategory={
                            CODE_PRESET_CATEGORY.backtest_close_long_ccond
                          }
                          chatGptLink={COPILOT_LINKS.trade_criteria}
                        />
                      );
                    }}
                  </Field>
                </div>

                <div style={{ marginTop: "16px" }}>
                  <Field name={formKeys.useShorts}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={BACKTEST_FORM_LABELS.is_short_selling_strategy}
                        >
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue(
                                formKeys.useShorts,
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
