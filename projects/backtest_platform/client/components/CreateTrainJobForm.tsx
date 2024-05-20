/* eslint-disable */
import React from "react";
import { useModelQuery } from "../clients/queries/queries";
import { usePathParams } from "../hooks/usePathParams";
import {
  Checkbox,
  NumberInput,
  NumberInputField,
  Spinner,
  useToast,
} from "@chakra-ui/react";
import { Field, Form, Formik } from "formik";
import { WithLabel } from "./form/WithLabel";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { createTrainJob } from "../clients/requests";

interface RouteParams {
  datasetName: string;
  modelId: string;
}

interface Props {
  onClose?: () => void;
}

export interface TrainJobForm {
  numEpochs: number;
  saveModelEveryEpoch: boolean;
  backtestOnValidationSet: boolean;
}

export const CreateTrainJobForm = ({ onClose }: Props) => {
  const { modelId } = usePathParams<RouteParams>();
  const { data } = useModelQuery(modelId);
  const toast = useToast();

  if (!data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const onSubmit = async (values: TrainJobForm) => {
    const res = await createTrainJob(modelId, {
      num_epochs: values.numEpochs,
      save_model_after_every_epoch: values.saveModelEveryEpoch,
      backtest_on_val_set: values.backtestOnValidationSet,
    });

    if (res.status === 200) {
      toast({
        title: "Started training",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      if (onClose) onClose();
    }
  };

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
        }}
      >
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
          <FormSubmitBar
            style={{ marginTop: "16px" }}
            cancelCallback={onClose}
          />
        </Form>
      </Formik>
    </div>
  );
};
