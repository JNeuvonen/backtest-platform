/* eslint-disable */
import React from "react";
import { useModelQuery } from "../clients/queries/queries";
import { usePathParams } from "../hooks/usePathParams";
import {
  Checkbox,
  NumberInput,
  NumberInputField,
  Spinner,
} from "@chakra-ui/react";
import { Field, Form, Formik } from "formik";
import { WithLabel } from "./form/WithLabel";
import { CodeEditor } from "./CodeEditor";
import { CodeHelper } from "../utils/constants";
import { FormSubmitBar } from "./form/FormSubmitBar";

interface RouteParams {
  datasetName: string;
  modelName: string;
}

interface Props {
  onClose?: () => void;
}

interface TrainJobForm {
  numEpochs: number;
  saveModelEveryEpoch: boolean;
  backtestOnValidationSet: boolean;
  enterTradeCriteria: string;
  exitTradeCriteria: string;
}

const generateEnterTradeExample = () => {
  const code = new CodeHelper();
  code.appendLine("def get_enter_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction > 1.01");
  return code.get();
};

const generateExitTradeExample = () => {
  const code = new CodeHelper();
  code.appendLine("def get_exit_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction < 0.99");
  return code.get();
};

export const CreateTrainJobForm = ({ onClose }: Props) => {
  const { modelName } = usePathParams<RouteParams>();
  const { data } = useModelQuery(modelName);

  if (!data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const onSubmit = async (values: TrainJobForm) => {};

  return (
    <div>
      <Formik
        onSubmit={(values: TrainJobForm) => {
          onSubmit(values);
        }}
        initialValues={{
          numEpochs: 100,
          saveModelEveryEpoch: false,
          backtestOnValidationSet: true,
          enterTradeCriteria: generateEnterTradeExample(),
          exitTradeCriteria: generateExitTradeExample(),
        }}
      >
        {({ values }) => (
          <Form>
            <WithLabel
              label="Number of epochs"
              containerStyles={{ maxWidth: "250px" }}
            >
              <Field name="numEpochs">
                {({ field }) => (
                  <NumberInput step={5} min={10} {...field}>
                    <NumberInputField {...field} />
                  </NumberInput>
                )}
              </Field>
            </WithLabel>

            <div style={{ marginTop: "16px" }}>
              <Field type="checkbox" name="saveModelEveryEpoch">
                {({ field }) => (
                  <Checkbox {...field} isChecked={field.value}>
                    Save model after every epoch
                  </Checkbox>
                )}
              </Field>
            </div>
            <div style={{ marginTop: "4px" }}>
              <Field type="checkbox" name="backtestOnValidationSet">
                {({ field }) => (
                  <Checkbox {...field} isChecked={field.value}>
                    Backtest on validation set
                  </Checkbox>
                )}
              </Field>
            </div>

            {values.backtestOnValidationSet && (
              <>
                <WithLabel
                  label="Enter trade criteria"
                  containerStyles={{ marginTop: "16px" }}
                >
                  <Field name="enterTradeCriteria">
                    {({ field, form }) => (
                      <CodeEditor
                        code={field.value}
                        setCode={(code) => form.setFieldValue(field.name, code)}
                        fontSize={15}
                        height={"200px"}
                        autoFocus={false}
                      />
                    )}
                  </Field>
                </WithLabel>
                <WithLabel
                  label="Exit trade criteria"
                  containerStyles={{ marginTop: "16px" }}
                >
                  <Field name="exitTradeCriteria">
                    {({ field, form }) => (
                      <CodeEditor
                        code={field.value}
                        setCode={(code) => form.setFieldValue(field.name, code)}
                        fontSize={15}
                        height={"200px"}
                        autoFocus={false}
                      />
                    )}
                  </Field>
                </WithLabel>
              </>
            )}

            <FormSubmitBar
              style={{ marginTop: "16px" }}
              cancelCallback={onClose}
            />
          </Form>
        )}
      </Formik>
    </div>
  );
};
