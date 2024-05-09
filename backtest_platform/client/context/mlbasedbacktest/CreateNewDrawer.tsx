import React, { useRef, useState } from "react";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { useMLBasedBacktestContext } from ".";
import { usePathParams } from "../../hooks/usePathParams";
import {
  useDatasetModelsQuery,
  useEpochValPredictions,
  useModelTrainMetadata,
  useTrainJobDetailed,
} from "../../clients/queries/queries";
import {
  Button,
  NumberInput,
  NumberInputField,
  Spinner,
  Switch,
  useToast,
} from "@chakra-ui/react";
import { Field, Form, Formik, FormikProps } from "formik";
import { DISK_KEYS, DiskManager } from "../../utils/disk";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";
import { BACKTEST_FORM_LABELS } from "../../utils/backtest";
import {
  getBacktestFormDefaultKeys,
  getBacktestFormDefaults,
} from "../../utils/backtest";
import { BUTTON_VARIANTS } from "../../theme";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../components/SelectFilter";
import { MultiValue, SingleValue } from "react-select";
import { GenericBarChart } from "../../components/charts/BarChart";
import { getNormalDistributionItems } from "../../utils/number";
import { ChakraSlider } from "../../components/chakra/Slider";
import { CodeEditor } from "../../components/CodeEditor";
import { CODE_PRESET_CATEGORY } from "../../utils/constants";
import { ChakraNumberStepper } from "../../components/ChakraNumberStepper";
import { ML_SIM_ENTER_DEFAULT, ML_SIM_EXIT_DEFAULT } from "../../utils/code";
import { ValidationSplitSlider } from "../../components/ValidationSplitSlider";
import { createMlBasedBacktest } from "../../clients/requests";
import { GenericAreaChart } from "../../components/charts/AreaChart";
import { getDataForSortedPredictions } from "../../pages/data/model/trainjob/info";

const formDiskManager = new DiskManager(DISK_KEYS.ml_backtest_form);

export interface MLBasedBacktestFormValues {
  use_latest_data: boolean;
  model: SingleValue<OptionType>;
  train_run: SingleValue<OptionType>;
  epoch: null | number;
  open_trade_code: string;
  close_trade_code: string;
  use_shorts: boolean;
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
  backtestName: string;
}

const getFormInitialValues = () => {
  const prevForm = formDiskManager.read();

  if (prevForm === null) {
    return {
      use_latest_data: false,
      model: null,
      train_run: null,
      epoch: null,
      open_trade_code: ML_SIM_ENTER_DEFAULT(),
      close_trade_code: ML_SIM_EXIT_DEFAULT(),
      use_shorts: true,
      ...getBacktestFormDefaults(),
    };
  }

  return {
    ...prevForm,
  };
};

const formKeys = {
  use_latest_data: "use_latest_data",
  model: "model",
  train_run: "train_run",
  epoch: "epoch",
  open_trade_code: "open_trade_code",
  close_trade_code: "close_trade_code",
  use_shorts: "use_shorts",
  ...getBacktestFormDefaultKeys(),
};

export const CreateNewMLBasedBacktestDrawer = () => {
  const { createNewDrawer } = useMLBasedBacktestContext();
  const { datasetName } = usePathParams<{ datasetName: string }>();
  const datasetModelsQuery = useDatasetModelsQuery(datasetName);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedTrainJobId, setSelectedTrainJobId] = useState<
    number | undefined
  >(undefined);
  const [selectedEpoch, setSelectedEpoch] = useState<number | undefined>(1);

  const modelTrainsQuery = useModelTrainMetadata(selectedModel);
  const trainRunQuery = useTrainJobDetailed(selectedTrainJobId);
  const epochPredsQuery = useEpochValPredictions(
    selectedTrainJobId,
    selectedEpoch
  );

  const formikRef = useRef<FormikProps<any>>(null);
  const toast = useToast();

  const onSubmit = async (values: MLBasedBacktestFormValues) => {
    if (!values.model || !values.train_run) return;
    const body = {
      backtest_data_range: values.backtestDataRange,
      fetch_latest_data: values.use_latest_data,
      dataset_name: datasetName,
      id_of_model: Number(values.model.value),
      train_run_id: Number(values.train_run.value),
      epoch: values.epoch,
      enter_trade_cond: values.open_trade_code,
      exit_trade_cond: values.close_trade_code,
      allow_shorts: values.use_shorts,
      use_time_based_close: values.useTimeBasedClose,
      use_profit_based_close: values.useProfitBasedClose,
      use_stop_loss_based_close: values.useStopLossBasedClose,
      trading_fees_perc: values.tradingFees,
      slippage_perc: values.slippage,
      short_fee_hourly: values.shortFeeHourly,
      take_profit_threshold_perc: values.takeProfitThresholdPerc,
      stop_loss_threshold_perc: values.stopLossThresholdPerc,
      name: values.backtestName,
      klines_until_close: values.klinesUntilClose,
    };

    const res = await createMlBasedBacktest(body);

    if (res.status === 200) {
      toast({
        title: "Backtest started",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      formDiskManager.save(values);
      createNewDrawer.onClose();
    }
  };

  let epochPredictions: number[] =
    epochPredsQuery.data && epochPredsQuery.data?.length > 0
      ? epochPredsQuery.data.map((item) => {
          return item["prediction"];
        })
      : [];

  if (!datasetModelsQuery.data) {
    return (
      <div>
        <ChakraDrawer
          title="Create a new backtest"
          drawerContentStyles={{ maxWidth: "80%" }}
          {...createNewDrawer}
        >
          <Spinner />
        </ChakraDrawer>
      </div>
    );
  }

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
                  <div style={{ width: "225px" }}>
                    <WithLabel label={"Select model"}>
                      <Field
                        name={formKeys.model}
                        as={SelectWithTextFilter}
                        options={datasetModelsQuery.data.map((item) => {
                          return {
                            value: item.id,
                            label: item.model_name,
                          };
                        })}
                        isMulti={false}
                        value={values[formKeys.model]}
                        placeholder={""}
                        onChange={(
                          selectedOptions:
                            | SingleValue<OptionType>
                            | MultiValue<OptionType>
                        ) => {
                          const option =
                            selectedOptions as SingleValue<OptionType>;
                          setFieldValue(formKeys.model, option);
                          setSelectedModel(option?.value as string);
                        }}
                      />
                    </WithLabel>
                  </div>

                  {modelTrainsQuery.data &&
                    modelTrainsQuery.data.length > 0 && (
                      <div style={{ width: "225px" }}>
                        <WithLabel label={"Select training run"}>
                          <Field
                            name={formKeys.train_run}
                            as={SelectWithTextFilter}
                            options={modelTrainsQuery.data.map((item, idx) => {
                              return {
                                value: item.train.id,
                                label: idx + 1,
                              };
                            })}
                            isMulti={false}
                            value={values[formKeys.train_run]}
                            placeholder={""}
                            onChange={(
                              selectedOptions:
                                | SingleValue<OptionType>
                                | MultiValue<OptionType>
                            ) => {
                              const option =
                                selectedOptions as SingleValue<OptionType>;
                              setFieldValue(formKeys.train_run, option);
                              setSelectedTrainJobId(Number(option?.value));
                            }}
                          />
                        </WithLabel>
                      </div>
                    )}
                  {trainRunQuery.data &&
                    trainRunQuery.data.epochs.length > 0 && (
                      <div style={{ width: "225px" }}>
                        <ChakraSlider
                          label={`Epoch number: ${values[formKeys.epoch]}`}
                          containerStyles={{ maxWidth: "300px" }}
                          min={1}
                          max={trainRunQuery.data.epochs.length}
                          onChange={(newValue: number) => {
                            setFieldValue(formKeys.epoch, newValue);
                            setSelectedEpoch(newValue);
                          }}
                          defaultValue={1}
                          value={values[formKeys.epoch]}
                        />
                      </div>
                    )}
                </div>
                {selectedEpoch !== null && (
                  <WithLabel
                    label="Validation predictions normal distribution"
                    containerStyles={{ marginTop: "16px" }}
                  >
                    <GenericBarChart
                      data={getNormalDistributionItems(epochPredictions)}
                      yAxisKey="count"
                      xAxisKey="label"
                      containerStyles={{ marginTop: "16px" }}
                    />
                  </WithLabel>
                )}

                {selectedEpoch !== null && (
                  <WithLabel
                    label="Validation predictions sorted by smallest first"
                    containerStyles={{ marginTop: "32px" }}
                  >
                    <GenericAreaChart
                      data={getDataForSortedPredictions(epochPredictions)}
                      yAxisKey="prediction"
                      xAxisKey="num"
                      containerStyles={{ marginTop: "16px" }}
                    />
                  </WithLabel>
                )}

                <div>
                  <Field name={formKeys.open_trade_code}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(
                              formKeys.open_trade_code,
                              newState
                            )
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={BACKTEST_FORM_LABELS.long_condition}
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
                  <Field name={formKeys.close_trade_code}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(
                              formKeys.close_trade_code,
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
                        />
                      );
                    }}
                  </Field>
                </div>
                <div
                  style={{ marginTop: "16px", display: "flex", gap: "16px" }}
                >
                  <Field name={formKeys.use_latest_data}>
                    {({ field, form }) => {
                      return (
                        <WithLabel label={"Download latest data"}>
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue(
                                formKeys.use_latest_data,
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
                  <Field name={formKeys.use_shorts}>
                    {({ field, form }) => {
                      return (
                        <WithLabel label={"Allow short selling"}>
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue(
                                formKeys.use_shorts,
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
