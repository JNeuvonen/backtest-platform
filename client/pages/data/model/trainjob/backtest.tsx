import React from "react";
import {
  useDatasetsQuery,
  useTrainJobDetailed,
} from "../../../../clients/queries/queries";
import { usePathParams } from "../../../../hooks/usePathParams";
import { Button, Spinner } from "@chakra-ui/react";
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
  const { data } = useTrainJobDetailed(trainJobId);
  const { data: allDatasets } = useDatasetsQuery();

  if (!data || !allDatasets) {
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
      console.log(res);
    } else {
      console.log("fail: ", res);
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
            <Field name="selectedModel">
              {({ field }) => (
                <ChakraSelect
                  label="Select Model"
                  options={generateSelectWeightsOptions(data.epochs)}
                  onChange={(value) =>
                    field.onChange({ target: { name: "selectedModel", value } })
                  }
                  containerStyle={{ width: "350px" }}
                />
              )}
            </Field>

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
