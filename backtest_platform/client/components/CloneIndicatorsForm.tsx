import React from "react";
import {
  useBinanceTickersQuery,
  useDatasetsQuery,
} from "../clients/queries/queries";
import { Select, Spinner, Switch } from "@chakra-ui/react";
import { SelectDatasetV2 } from "./SelectDatasetV2";
import { Field, Form, Formik } from "formik";
import { DatasetMetadata } from "../clients/queries/response-types";
import { WithLabel } from "./form/WithLabel";
import { OptionType, SelectWithTextFilter } from "./SelectFilter";
import { binanceTickSelectOptions } from "../pages/Datasets";
import { MultiValue } from "react-select";
import { GET_KLINE_OPTIONS } from "../utils/constants";
import { FormSubmitBar } from "./form/FormSubmitBar";

const formKeys = {
  datasets: "datasets",
  newDatasets: "newDatasets",
  candleInterval: "candleInterval",
  useFutures: "useFutures",
};

interface Props {
  cancelCallback: () => void;
  submitCallback: () => void;
}

export const CloneIndicatorsDrawer = ({ cancelCallback }: Props) => {
  const datasetsQuery = useDatasetsQuery();
  const binanceTickersQuery = useBinanceTickersQuery();

  if (!datasetsQuery.data || !binanceTickersQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }
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
          console.log(values);
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
