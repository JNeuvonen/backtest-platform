import React from "react";
import { ChakraModal } from "../../components/chakra/modal";
import { useBacktestContext } from ".";
import { Field, Form, Formik } from "formik";
import { FormControl, FormLabel, Select, Spinner } from "@chakra-ui/react";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../components/SelectFilter";
import { binanceTickSelectOptions } from "../../pages/Datasets";
import { MultiValue } from "react-select";
import { useBinanceTickersQuery } from "../../clients/queries/queries";
import { GET_KLINE_OPTIONS } from "../../utils/constants";

const formKeys = {
  pairs: "pairs",
  interval: "interval",
};

const getFormInitialValues = () => {
  return {
    pairs: [],
    interval: "1h",
  };
};

export interface MassBacktest {
  pairs: string[];
  interval: string;
}

export const BacktestOnManyPairs = () => {
  const { runBacktestOnManyPairsModal } = useBacktestContext();
  const binanceTickersQuery = useBinanceTickersQuery();

  if (binanceTickersQuery.isLoading || !binanceTickersQuery.data) {
    return (
      <ChakraModal
        {...runBacktestOnManyPairsModal}
        title="Run python"
        modalContentStyle={{ maxWidth: "60%" }}
      >
        <Spinner />
      </ChakraModal>
    );
  }

  const onSubmit = async (formValues: MassBacktest) => {};

  return (
    <ChakraModal
      {...runBacktestOnManyPairsModal}
      title="Run python"
      modalContentStyle={{ maxWidth: "60%" }}
    >
      <Formik initialValues={getFormInitialValues()} onSubmit={onSubmit}>
        {({ setFieldValue }) => {
          return (
            <Form>
              <FormControl>
                <FormLabel fontSize={"x-large"}>
                  Pairs to be downloaded
                </FormLabel>
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
                />
              </FormControl>
              <FormControl marginTop={"8px"}>
                <FormLabel fontSize={"x-large"}>Candle interval</FormLabel>
                <Field name={formKeys.interval} as={Select}>
                  {GET_KLINE_OPTIONS().map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </Field>
              </FormControl>
            </Form>
          );
        }}
      </Formik>
    </ChakraModal>
  );
};
