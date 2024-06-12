import { Switch, UseDisclosureReturn } from "@chakra-ui/react";
import React from "react";
import { ChakraDrawer } from "./chakra/Drawer";
import { Field, Form, Formik } from "formik";
import { WithLabel } from "./form/WithLabel";
import { ChakraInput } from "./chakra/input";
import { BACKTEST_FORM_LABELS } from "../utils/backtest";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";

interface Props {
  drawerControls: UseDisclosureReturn;
}

const formKeys = {
  backtestName: "backtestName",
  useLatestData: "useLatestData",
  startDate: "startDate",
  endDate: "endDate",
};

const getFormInitialValues = () => {
  return {
    backtestName: "",
    useLatestData: false,
    startDate: null,
    endDate: null,
  };
};

const FormikDatePicker = ({ field, form, ...props }) => {
  return (
    <DatePicker
      {...field}
      {...props}
      selected={(field.value && new Date(field.value)) || null}
      onChange={(date) => form.setFieldValue(field.name, date)}
    />
  );
};

export const CreateMassRuleBasedSim = ({ drawerControls }: Props) => {
  return (
    <div>
      <ChakraDrawer
        title={"Create a new backtest"}
        drawerContentStyles={{ maxWidth: "80%" }}
        {...drawerControls}
      >
        <Formik
          onSubmit={(values) => {}}
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
              <div style={{ marginTop: "16px", display: "flex", gap: "16px" }}>
                <div>
                  <Field name={formKeys.useLatestData}>
                    {({ field, form }) => {
                      return (
                        <WithLabel label={"Download latest data"}>
                          <Switch
                            isChecked={field.value}
                            onChange={() =>
                              form.setFieldValue(
                                formKeys.useLatestData,
                                !field.value
                              )
                            }
                          />
                        </WithLabel>
                      );
                    }}
                  </Field>
                </div>
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
            </Form>
          )}
        </Formik>
      </ChakraDrawer>
    </div>
  );
};
