import React from "react";
import {
  useBinanceTickersQuery,
  useDatasetsQuery,
} from "../clients/queries/queries";
import { Select, Spinner, Switch, useToast } from "@chakra-ui/react";
import { SelectDatasetV2 } from "./SelectDatasetV2";
import { Field, Form, Formik } from "formik";
import { DatasetMetadata } from "../clients/queries/response-types";
import { WithLabel } from "./form/WithLabel";
import { OptionType, SelectWithTextFilter } from "./SelectFilter";
import { binanceTickSelectOptions } from "../pages/Datasets";
import { MultiValue } from "react-select";
import { GET_KLINE_OPTIONS } from "../utils/constants";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { cloneIndicators } from "../clients/requests";

const formKeys = {
  datasets: "datasets",
  newDatasets: "newDatasets",
  candleInterval: "candleInterval",
  useFutures: "useFutures",
};

export interface CloneIndicatorsFormValues {
  datasets: OptionType[];
  newDatasets: OptionType[];
  candleInterval: string | null;
  useFutures: boolean | null;
}

export interface CloneIndicatorsBodyBackendFormat {
  existing_datasets: string[];
  new_datasets: string[];
  candle_interval: string | null;
  use_futures: boolean | null;
}

interface Props {
  datasetName: string;
  cancelCallback: () => void;
  submitCallback: () => void;
}

export const CloneIndicatorsDrawer = ({
  datasetName,
  cancelCallback,
  submitCallback,
}: Props) => {
  const datasetsQuery = useDatasetsQuery();
  const binanceTickersQuery = useBinanceTickersQuery();
  const toast = useToast();

  if (!datasetsQuery.data || !binanceTickersQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const submit = async (values: CloneIndicatorsFormValues) => {
    const res = await cloneIndicators(datasetName, {
      existing_datasets: values.datasets.map((item) => item.value),
      new_datasets: values.newDatasets.map((item) => item.value),
      candle_interval: values.candleInterval,
      use_futures: values.useFutures,
    });

    if (res.status === 200) {
      toast({
        title: "Cloned indicators",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      submitCallback();
    }
  };

  return (
    <div>
      <Formik
        initialValues={{
          datasets: [],
          newDatasets: [],
          candleInterval: null,
          useFutures: false,
        }}
        onSubmit={(values) => {
          submit(values);
        }}
      >
        {({ values, setFieldValue }) => (
          <Form>
            <div>
              <Field
                name={formKeys.datasets}
                component={({ field, form }) => (
                  <WithLabel label={"Select dataset"}>
                    <SelectDatasetV2
                      datasets={datasetsQuery.data as DatasetMetadata[]}
                      onSelect={(newValues) => {
                        form.setFieldValue(field.name, newValues);
                      }}
                      multiSelect={true}
                      value={values.datasets}
                    />
                  </WithLabel>
                )}
              />
            </div>
            <div style={{ marginTop: "16px" }}>
              <WithLabel label={"New datasets"}>
                <Field
                  name="selectedTickers"
                  as={SelectWithTextFilter}
                  options={binanceTickSelectOptions(
                    binanceTickersQuery.data.res.pairs
                  )}
                  onChange={(selectedOptions: MultiValue<OptionType>) =>
                    setFieldValue(formKeys.newDatasets, selectedOptions)
                  }
                  isMulti={true}
                  closeMenuOnSelect={false}
                />
              </WithLabel>
            </div>

            {values.newDatasets.length > 0 && (
              <>
                <div style={{ marginTop: "16px" }}>
                  <WithLabel label={"Candle interval"}>
                    <Field name={formKeys.candleInterval} as={Select}>
                      {GET_KLINE_OPTIONS().map((item) => (
                        <option key={item} value={item}>
                          {item}
                        </option>
                      ))}
                    </Field>
                  </WithLabel>
                </div>
                <div>
                  <Field
                    name={formKeys.datasets}
                    component={({ form }) => (
                      <WithLabel
                        label={"Use futures data"}
                        containerStyles={{ marginTop: "8px" }}
                      >
                        <Switch
                          isChecked={values.useFutures}
                          onChange={() =>
                            form.setFieldValue(
                              formKeys.useFutures,
                              !values.useFutures
                            )
                          }
                        />
                      </WithLabel>
                    )}
                  />
                </div>
              </>
            )}

            <FormSubmitBar
              style={{ marginTop: "32px" }}
              cancelCallback={cancelCallback}
            />
          </Form>
        )}
      </Formik>
    </div>
  );
};
