import React, { useState } from "react";
import {
  useDatasetsQuery,
  useTrainJobBacktests,
  useTrainJobDetailed,
} from "../../../../clients/queries/queries";
import { usePathParams } from "../../../../hooks/usePathParams";
import { Button, Heading, Spinner, Stack, useToast } from "@chakra-ui/react";
import {
  ChakraSelect,
  SelectOption,
} from "../../../../components/chakra/select";
import { EpochInfo } from "../../../../clients/queries/response-types";
import { roundNumberDropRemaining } from "../../../../utils/number";
import { Formik, Form, Field } from "formik";
import { CodeEditor } from "../../../../components/CodeEditor";
import { runBacktest } from "../../../../clients/requests";
import { WithLabel } from "../../../../components/form/WithLabel";
import { PredAndPriceChart } from "../../../../components/charts/PredAndPriceChart";
import { SelectColumnPopover } from "../../../../components/SelectTargetColumnPopover";
import {
  ML_ENTER_TRADE_COND,
  ML_EXIT_TRADE_COND,
} from "../../../../utils/code";

export interface BacktestForm {
  selectedModel: string | null;
  priceColumn: string;
  exitTradeCond: string;
  enterTradeCond: string;
}

export const BacktestModelPage = () => {
  const { trainJobId } = usePathParams<{
    trainJobId: string;
    datasetName?: string;
  }>();
  const { data: modelDataDetailed, refetch: refetchTrainjob } =
    useTrainJobDetailed(trainJobId);
  const { data: allDatasets, refetch: refetchDatasets } = useDatasetsQuery();
  const { refetch: refetchBacktests } = useTrainJobBacktests(trainJobId);
  const toast = useToast();
  const [epochNr, setEpochNr] = useState("");

  if (!modelDataDetailed || !allDatasets) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const generateSelectWeightsOptions = (epochs: EpochInfo[]) => {
    const ret = [] as SelectOption[];
    for (let i = 0; i < epochs.length; i++) {
      const item = epochs[i];
      ret.push({
        label: `Epoch ${item.epoch}, Train loss: ${roundNumberDropRemaining(
          item.train_loss,
          3
        )}, Val: loss ${roundNumberDropRemaining(item.val_loss, 3)}`,
        value: String(item.epoch),
      });
    }
    return ret;
  };

  const handleSubmit = async (values: BacktestForm) => {
    const payload = {
      epoch_nr: Number(values.selectedModel),
      enter_trade_cond: values.enterTradeCond,
      exit_trade_cond: values.exitTradeCond,
      price_col: values.priceColumn,
      dataset_name: modelDataDetailed.dataset_metadata.dataset_name,
    };

    const res = await runBacktest(trainJobId, payload);

    if (res.status === 200) {
      refetchTrainjob();
      refetchDatasets();
      refetchBacktests();
      toast({
        title: "Finished running the backtest.",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
    }
  };

  function isFormReady(values: BacktestForm) {
    const isReady =
      values.priceColumn &&
      values.selectedModel &&
      values.exitTradeCond &&
      values.enterTradeCond;
    return isReady;
  }

  return (
    <div>
      <Formik
        initialValues={{
          selectedModel: null,
          exitTradeCond: ML_EXIT_TRADE_COND(),
          priceColumn: modelDataDetailed.dataset_metadata.price_column,
          enterTradeCond: ML_ENTER_TRADE_COND(),
        }}
        onSubmit={handleSubmit}
      >
        {({ values }) => (
          <Form>
            <Heading
              size={"md"}
              marginTop={"8px"}
            >{`Dataset: ${modelDataDetailed.dataset_metadata.dataset_name}`}</Heading>
            <Stack direction="row" marginTop={"16px"}>
              <Field name="selectedModel">
                {({ field }) => (
                  <ChakraSelect
                    label="Select Model"
                    options={generateSelectWeightsOptions(
                      modelDataDetailed.epochs
                    )}
                    onChange={(value) => {
                      field.onChange({
                        target: { name: "selectedModel", value },
                      });
                      setEpochNr(value);
                    }}
                    containerStyle={{ width: "350px" }}
                  />
                )}
              </Field>
              <Field>
                {({ field }) => {
                  return (
                    <WithLabel
                      label="Price column"
                      containerStyles={{ width: "350px" }}
                    >
                      <SelectColumnPopover
                        options={
                          modelDataDetailed.dataset_columns
                            ? modelDataDetailed.dataset_columns.map((item) => {
                                return {
                                  label: item,
                                  value: item,
                                };
                              })
                            : []
                        }
                        placeholder={field.value.priceColumn}
                        selectCallback={(value: string) => {
                          field.onChange({
                            target: { name: "priceColumn", value },
                          });
                        }}
                      />
                    </WithLabel>
                  );
                }}
              </Field>
            </Stack>
            {epochNr && (
              <PredAndPriceChart
                kline_open_times={
                  modelDataDetailed.train_job.backtest_kline_open_times
                }
                _prices={modelDataDetailed.train_job.backtest_prices}
                epoch={modelDataDetailed.epochs[epochNr]}
              />
            )}
            <div style={{ marginTop: "16px" }}>
              <Field name="enterTradeCond">
                {({ field, form }) => {
                  return (
                    <CodeEditor
                      code={field.value}
                      setCode={(newCode) =>
                        form.setFieldValue("enterAndExitCriteria", newCode)
                      }
                      height="200px"
                      label="Enter trade criteria"
                      fontSize={14}
                    />
                  );
                }}
              </Field>
            </div>

            <div style={{ marginTop: "16px" }}>
              <Field name="exitTradeCond">
                {({ field, form }) => {
                  return (
                    <CodeEditor
                      code={field.value}
                      setCode={(newCode) =>
                        form.setFieldValue("enterAndExitCriteria", newCode)
                      }
                      height="200px"
                      label="Exit trade criteria"
                      fontSize={14}
                    />
                  );
                }}
              </Field>
            </div>

            <Button
              type="submit"
              marginTop={"16px"}
              isDisabled={!isFormReady(values)}
            >
              Run backtest
            </Button>
          </Form>
        )}
      </Formik>
    </div>
  );
};
