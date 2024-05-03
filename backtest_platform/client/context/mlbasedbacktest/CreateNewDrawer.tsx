import React, { useRef, useState } from "react";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { useMLBasedBacktestContext } from ".";
import { usePathParams } from "../../hooks/usePathParams";
import {
  useDatasetModelsQuery,
  useModelTrainMetadata,
  useTrainJobDetailed,
} from "../../clients/queries/queries";
import { Button, Spinner } from "@chakra-ui/react";
import { Field, Form, Formik, FormikProps } from "formik";
import { DISK_KEYS, DiskManager } from "../../utils/disk";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";
import { BACKTEST_FORM_LABELS } from "../../utils/backtest";
import { getBacktestFormDefaultKeys } from "../../utils/backtest";
import { BUTTON_VARIANTS } from "../../theme";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../components/SelectFilter";
import { MultiValue, SingleValue } from "react-select";
import { GenericBarChart } from "../../components/charts/BarChart";
import { getNormalDistributionItems } from "../../utils/number";
import { ChakraSlider } from "../../components/chakra/Slider";

const formDiskManager = new DiskManager(DISK_KEYS.ml_backtest_form);

const getFormInitialValues = () => {
  return {};
};

const formKeys = {
  buy_criteria: "buy_criteria",
  sell_criteria: "short_criteria",
  use_latest_data: "use_latest_data",
  model: "model",
  train_run: "train_run",
  epoch: "epoch",
  ...getBacktestFormDefaultKeys(),
};

export const CreateNewMLBasedBacktestDrawer = () => {
  const { createNewDrawer } = useMLBasedBacktestContext();
  const { datasetName } = usePathParams<{ datasetName: string }>();
  const datasetModelsQuery = useDatasetModelsQuery(datasetName);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedTrainJobId, setSelectedTrainJobId] = useState<number | null>(
    null
  );
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null);

  const modelTrainsQuery = useModelTrainMetadata(selectedModel);
  const trainRunQuery = useTrainJobDetailed(selectedTrainJobId);

  const formikRef = useRef<FormikProps<any>>(null);

  const parseEpochPredictions = (epochNr: number | null) => {
    if (!trainRunQuery.data || !epochNr) return [];
    return trainRunQuery.data.epochs.length > 0
      ? JSON.parse(trainRunQuery.data.epochs[epochNr - 1].val_predictions).map(
          (item: number[]) => item[0]
        )
      : [];
  };
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
            onSubmit={() => {}}
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
                            value: item.model_name,
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

                <WithLabel
                  label="Validation predictions normal distribution"
                  containerStyles={{ marginTop: "16px" }}
                >
                  <GenericBarChart
                    data={getNormalDistributionItems(
                      parseEpochPredictions(selectedEpoch as number)
                    )}
                    yAxisKey="count"
                    xAxisKey="label"
                    containerStyles={{ marginTop: "16px" }}
                  />
                </WithLabel>
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
