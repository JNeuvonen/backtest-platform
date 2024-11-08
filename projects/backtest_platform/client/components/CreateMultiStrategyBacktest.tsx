import { UseDisclosureReturn, useDisclosure } from "@chakra-ui/react";
import React from "react";
import { ChakraDrawer } from "./chakra/Drawer";
import { Field, Form, Formik } from "formik";
import { WithLabel } from "./form/WithLabel";
import { ChakraInput } from "./chakra/input";
import { BACKTEST_FORM_LABELS } from "../utils/backtest";
import { SelectMassSimStrategies } from "./SelectMassSimStrategies";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { FormikDatePicker } from "./FormikDataPicker";

export interface MultiStrategyBacktest {
  backtestName: string;
  selectedStrategyIds: number[];
  startDate: Date | null;
  endDate: Date | null;
}

interface Props {
  drawerControls: UseDisclosureReturn;
  onSubmit: (values: MultiStrategyBacktest) => void;
}

const formKeys = {
  backtestName: "backtestName",
  selectedStrategyIds: "selectedStrategyIds",
  startDate: "startDate",
  endDate: "endDate",
};

const getFormInitialValues = () => {
  return {
    [formKeys.backtestName]: "",
    [formKeys.selectedStrategyIds]: [] as number[],
    [formKeys.startDate]: null,
    [formKeys.endDate]: null,
  };
};

export const CreateMultiStrategyBacktest = ({
  drawerControls,
  onSubmit,
}: Props) => {
  return (
    <div>
      <ChakraDrawer
        title={"Create new multistrategy backtest"}
        drawerContentStyles={{ maxWidth: "80%" }}
        {...drawerControls}
      >
        <Formik
          onSubmit={(values) => {
            onSubmit(values);
          }}
          initialValues={getFormInitialValues()}
        >
          {({ values }) => (
            <Form>
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
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  marginTop: "16px",
                }}
              >
                <div>
                  <WithLabel label={"Start date"}>
                    <Field
                      name={formKeys.startDate}
                      component={FormikDatePicker}
                    />
                  </WithLabel>
                </div>
                <div>
                  <WithLabel label={"End date"}>
                    <Field
                      name={formKeys.endDate}
                      component={FormikDatePicker}
                    />
                  </WithLabel>
                </div>
              </div>
              <div style={{ marginTop: "8px" }}>
                <Field name={formKeys.selectedStrategyIds}>
                  {({ form, field }) => {
                    return (
                      <SelectMassSimStrategies
                        selectedIds={field.value}
                        onSelect={(selectedIds) => {
                          form.setFieldValue(
                            formKeys.selectedStrategyIds,
                            selectedIds
                          );
                        }}
                      />
                    );
                  }}
                </Field>
              </div>

              <div style={{ marginTop: "16px" }}>
                <FormSubmitBar cancelCallback={drawerControls.onClose} />
              </div>
            </Form>
          )}
        </Formik>
      </ChakraDrawer>
    </div>
  );
};
