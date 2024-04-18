import React from "react";
import { ChakraModal } from "../../components/chakra/modal";
import { useBacktestContext } from ".";
import { Field, Form, Formik } from "formik";
import {
  Button,
  FormControl,
  FormLabel,
  Select,
  Spinner,
  useToast,
} from "@chakra-ui/react";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../components/SelectFilter";
import { binanceTickSelectOptions } from "../../pages/Datasets";
import { MultiValue } from "react-select";
import {
  useBacktestById,
  useBinanceTickersQuery,
} from "../../clients/queries/queries";
import { usePathParams } from "../../hooks/usePathParams";
import { postMassBacktest } from "../../clients/requests";

const formKeys = {
  pairs: "pairs",
};

const getFormInitialValues = () => {
  return {
    pairs: [] as MultiValue<OptionType>,
  };
};

interface PathParams {
  datasetName: string;
  backtestId: number;
}

export interface MassBacktest {
  pairs: MultiValue<OptionType>;
  interval: string;
}

export const BacktestOnManyPairs = () => {
  const { backtestId } = usePathParams<PathParams>();
  const { runBacktestOnManyPairsModal } = useBacktestContext();

  const binanceTickersQuery = useBinanceTickersQuery();
  const backtestQuery = useBacktestById(Number(backtestId));

  const toast = useToast();

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

  const onSubmit = async (formValues: MassBacktest) => {
    if (!backtestQuery.data?.data.id) return;

    const symbols = formValues.pairs.map((item) => item.label);

    const res = await postMassBacktest({
      symbols: symbols,
      original_backtest_id: backtestQuery.data?.data.id,
    });

    if (res.status === 200) {
      toast({
        title: "Started mass backtest",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
    }
  };

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
                  Select pairs for mass sim
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

              <Button type="submit" marginTop={"16px"}>
                Submit
              </Button>
            </Form>
          );
        }}
      </Formik>
    </ChakraModal>
  );
};
