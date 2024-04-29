import React, { useRef } from "react";
import { useMassBacktestContext } from ".";
import { ChakraDrawer } from "../../components/chakra/Drawer";
import { Field, Form, Formik, FormikProps } from "formik";
import {
  BACKTEST_FORM_LABELS,
  getBacktestFormDefaultKeys,
} from "../../utils/backtest";
import { DISK_KEYS, DiskManager } from "../../utils/disk";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";
import { CodeEditor } from "../../components/CodeEditor";
import { CODE_PRESET_CATEGORY } from "../../utils/constants";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../components/SelectFilter";
import { binanceTickSelectOptions } from "../../pages/Datasets";
import { useBinanceTickersQuery } from "../../clients/queries/queries";
import { MultiValue } from "react-select";
import {
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Spinner,
} from "@chakra-ui/react";

const backtestDiskManager = new DiskManager(DISK_KEYS.mass_long_short_form);

const getFormInitialValues = () => {
  return {
    buy_criteria: "",
    short_criteria: "",
    exit_cond: "",
    pairs: [],
    max_simultaneous_positions: 15,
    max_leverage_ratio: 2.5,
  };
};

const formKeys = {
  buy_criteria: "buy_criteria",
  sell_criteria: "short_criteria",
  exit_cond: "exit_cond",
  pairs: "pairs",
  max_simultaneous_positions: "max_simultaneous_positions",
  max_leverage_ratio: "max_leverage_ratio",
  ...getBacktestFormDefaultKeys(),
};

export const BulkLongShortCreateNew = () => {
  const { createNewDrawer } = useMassBacktestContext();
  const binanceTickersQuery = useBinanceTickersQuery();
  const formikRef = useRef<FormikProps<any>>(null);

  if (!binanceTickersQuery.data) {
    return (
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...createNewDrawer}
      >
        <Spinner />
      </ChakraDrawer>
    );
  }

  return (
    <div>
      <ChakraDrawer
        title="Create a new backtest"
        drawerContentStyles={{ maxWidth: "80%" }}
        {...createNewDrawer}
      >
        <div>
          <Formik
            onSubmit={(values) => {}}
            initialValues={getFormInitialValues()}
            innerRef={formikRef}
            enableReinitialize
          >
            {({ values, setFieldValue }) => (
              <Form>
                <div
                  style={{
                    display: "flex",
                    alignItems: "end",
                    gap: "16px",
                  }}
                >
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
                  <div style={{ marginTop: "16px", width: "600px" }}>
                    <WithLabel label={"Select pairs"}>
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
                        value={values.pairs}
                      />
                    </WithLabel>
                  </div>
                </div>

                <div>
                  <Field name={formKeys.buy_criteria}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) => {
                            form.setFieldValue(formKeys.buy_criteria, newState);
                          }}
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={BACKTEST_FORM_LABELS.long_short_buy_condition}
                          codeContainerStyles={{ width: "100%" }}
                          height={"250px"}
                          presetCategory={
                            CODE_PRESET_CATEGORY.backtest_long_cond
                          }
                        />
                      );
                    }}
                  </Field>
                </div>
                <div>
                  <Field name={formKeys.sell_criteria}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(formKeys.sell_criteria, newState)
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={BACKTEST_FORM_LABELS.long_short_sell_condition}
                          codeContainerStyles={{ width: "100%" }}
                          height={"250px"}
                          presetCategory={
                            CODE_PRESET_CATEGORY.backtest_close_long_ccond
                          }
                        />
                      );
                    }}
                  </Field>
                </div>
                <div>
                  <Field name={formKeys.exit_cond}>
                    {({ field, form }) => {
                      return (
                        <CodeEditor
                          code={field.value}
                          setCode={(newState) =>
                            form.setFieldValue(formKeys.exit_cond, newState)
                          }
                          style={{ marginTop: "16px" }}
                          fontSize={13}
                          label={"Pair exit condition"}
                          codeContainerStyles={{ width: "100%" }}
                          height={"250px"}
                          presetCategory={
                            CODE_PRESET_CATEGORY.backtest_pair_trade_exit_cond
                          }
                        />
                      );
                    }}
                  </Field>
                </div>

                <div>
                  <Field name={formKeys.max_simultaneous_positions}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Max simultaneous positions"}
                          containerStyles={{
                            maxWidth: "300px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={1}
                            min={0}
                            value={field.value}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                formKeys.klinesUntilClose,
                                parseInt(valueString)
                              )
                            }
                          >
                            <NumberInputField />
                          </NumberInput>
                        </WithLabel>
                      );
                    }}
                  </Field>
                </div>
                <div>
                  <Field name={formKeys.max_leverage_ratio}>
                    {({ field, form }) => {
                      return (
                        <WithLabel
                          label={"Max leverage ratio"}
                          containerStyles={{
                            maxWidth: "200px",
                            marginTop: "16px",
                          }}
                        >
                          <NumberInput
                            step={0.1}
                            max={3.0}
                            min={0.0}
                            precision={2}
                            value={field.value}
                            onChange={(valueString) =>
                              form.setFieldValue(
                                formKeys.max_leverage_ratio,
                                parseFloat(valueString)
                              )
                            }
                          >
                            <NumberInputField />
                            <NumberInputStepper>
                              <NumberIncrementStepper color={"white"} />
                              <NumberDecrementStepper color={"white"} />
                            </NumberInputStepper>
                          </NumberInput>
                        </WithLabel>
                      );
                    }}
                  </Field>
                </div>
              </Form>
            )}
          </Formik>
        </div>
      </ChakraDrawer>
    </div>
  );
};
