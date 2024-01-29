import React, { useState } from "react";
import {
  useDatasetsQuery,
  useTrainJobBacktests,
  useTrainJobDetailed,
} from "../../../../clients/queries/queries";
import { usePathParams } from "../../../../hooks/usePathParams";
import { Button, Spinner, Stack, useToast } from "@chakra-ui/react";
import {
  ChakraSelect,
  SelectOption,
} from "../../../../components/chakra/select";
import { EpochInfo } from "../../../../clients/queries/response-types";
import { roundNumberDropRemaining } from "../../../../utils/number";
import { Formik, Form, Field } from "formik";
import { CodeEditor } from "../../../../components/CodeEditor";
import { CodeHelper } from "../../../../utils/constants";
import { runBacktest } from "../../../../clients/requests";
import { WithLabel } from "../../../../components/form/WithLabel";
import { BUTTON_VARIANTS } from "../../../../theme";
import { PredAndPriceChart } from "../../../../components/charts/PredAndPriceChart";

export interface BacktestForm {
  selectedModel: string | null;
  enterAndExitCriteria: string;
}

const getTradeCriteriaDefaultCode = () => {
  const code = new CodeHelper();
  code.appendLine("def get_enter_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction > 1.01");
  code.appendLine("");
  code.reduceIndent();

  code.appendLine("def get_exit_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction < 0.99");
  return code.get();
};

export const BacktestModelPage = () => {
  const { trainJobId } = usePathParams<{
    trainJobId: string;
    datasetName?: string;
  }>();
  const { data: trainJob, refetch: refetchTrainjob } =
    useTrainJobDetailed(trainJobId);
  const { data: allDatasets, refetch: refetchDatasets } = useDatasetsQuery();
  const { refetch: refetchBacktests } = useTrainJobBacktests(trainJobId);
  const toast = useToast();
  const [epochNr, setEpochNr] = useState("");

  if (!trainJob || !allDatasets) {
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
      enter_and_exit_criteria: values.enterAndExitCriteria,
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

  return (
    <div>
      <Formik
        initialValues={{
          selectedModel: null,
          enterAndExitCriteria: getTradeCriteriaDefaultCode(),
        }}
        onSubmit={handleSubmit}
      >
        {() => (
          <Form>
            <Stack direction="row">
              <Field name="selectedModel">
                {({ field }) => (
                  <ChakraSelect
                    label="Select Model"
                    options={generateSelectWeightsOptions(trainJob.epochs)}
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
            </Stack>
            {epochNr && (
              <WithLabel
                label="Price and prediction chart"
                containerStyles={{ marginTop: "16px" }}
              >
                <PredAndPriceChart
                  kline_open_times={
                    trainJob.train_job.backtest_kline_open_times
                  }
                  _prices={trainJob.train_job.backtest_prices}
                  epoch={trainJob.epochs[epochNr]}
                />
              </WithLabel>
            )}
            <div style={{ marginTop: "16px" }}>
              <Field name="enterAndExitCriteria">
                {({ field, form }) => {
                  return (
                    <CodeEditor
                      code={field.value}
                      setCode={(newCode) =>
                        form.setFieldValue("enterAndExitCriteria", newCode)
                      }
                      label="Criteria for exit and enter"
                      fontSize={14}
                    />
                  );
                }}
              </Field>
            </div>

            <Button type="submit" marginTop={"16px"}>
              Run backtest
            </Button>
          </Form>
        )}
      </Formik>
    </div>
  );
};
