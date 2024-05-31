import { NumberInput, NumberInputField, Switch } from "@chakra-ui/react";
import { Field, Form, Formik } from "formik";
import { ChakraNumberStepper } from "src/components/chakra";
import { FormSubmitBar } from "src/components/FormSubmitBar";
import { WithLabel } from "src/components/WithLabel";

const BULK_UPDATE_SYMBOLS_FORM_KEYS = {
  closeOnly: "closeOnly",
  shouldCloseTrade: "shouldCloseTrade",
  allocationPerSymbol: "allocationPerSymbol",
};

interface FormTypes {
  closeOnly: boolean;
  shouldCloseTrade: boolean;
  allocationPerSymbol: number;
}

interface Props {
  onSubmit: (values: FormTypes) => void;
  onClose: () => void;
}

export const BulkUpdateRowsForm = ({ onSubmit, onClose }: Props) => {
  return (
    <div>
      <Formik
        initialValues={{
          [BULK_UPDATE_SYMBOLS_FORM_KEYS.closeOnly]: false,
          [BULK_UPDATE_SYMBOLS_FORM_KEYS.shouldCloseTrade]: false,
          [BULK_UPDATE_SYMBOLS_FORM_KEYS.allocationPerSymbol]: 2,
        }}
        onSubmit={(values: FormTypes) => {
          onSubmit({
            closeOnly: values.closeOnly,
            shouldCloseTrade: values.shouldCloseTrade,
            allocationPerSymbol: values.allocationPerSymbol,
          });
          onClose();
        }}
      >
        <Form>
          <div>
            <Field
              name={BULK_UPDATE_SYMBOLS_FORM_KEYS.shouldCloseTrade}
              component={({ field, form }) => {
                return (
                  <WithLabel label={"Close only"}>
                    <Switch
                      isChecked={field.value}
                      onChange={() =>
                        form.setFieldValue(field.name, !field.value)
                      }
                    />
                  </WithLabel>
                );
              }}
            />
          </div>
          <div style={{ marginTop: "16px" }}>
            <Field
              name={BULK_UPDATE_SYMBOLS_FORM_KEYS.closeOnly}
              component={({ field, form }) => {
                return (
                  <WithLabel label={"Should close trade"}>
                    <Switch
                      isChecked={field.value}
                      onChange={() =>
                        form.setFieldValue(field.name, !field.value)
                      }
                    />
                  </WithLabel>
                );
              }}
            />
          </div>

          <div style={{ marginTop: "16px" }}>
            <Field
              name={BULK_UPDATE_SYMBOLS_FORM_KEYS.allocationPerSymbol}
              component={({ field, form }) => {
                return (
                  <WithLabel
                    label={"Allocation per symbol"}
                    containerStyles={{
                      maxWidth: "200px",
                    }}
                  >
                    <NumberInput
                      step={0.05}
                      min={0}
                      max={25}
                      value={field.value}
                      onChange={(value) => {
                        form.setFieldValue(field.name, parseFloat(value));
                      }}
                    >
                      <NumberInputField />
                      <ChakraNumberStepper />
                    </NumberInput>
                  </WithLabel>
                );
              }}
            />
          </div>
          <FormSubmitBar style={{ marginTop: "16px" }} />
        </Form>
      </Formik>
    </div>
  );
};
