import React from "react";
import { ChakraModal } from "../../components/chakra/modal";
import { useBacktestContext } from ".";
import { Field, Form, Formik } from "formik";
import {
  Button,
  FormControl,
  FormLabel,
  Spinner,
  Switch,
  Tooltip,
  useDisclosure,
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
import { WithLabel } from "../../components/form/WithLabel";
import { BUTTON_VARIANTS } from "../../theme";
import { ChakraPopover } from "../../components/chakra/popover";
import { BULK_SIM_PAIR_PRESETS } from "../../utils/hardcodedpresets";

const formKeys = {
  pairs: "pairs",
  useLatestData: "useLatestData",
};

const getFormInitialValues = () => {
  return {
    pairs: [] as MultiValue<OptionType>,
    useLatestData: true,
  };
};

interface PathParams {
  datasetName: string;
  backtestId: number;
}

export interface MassBacktest {
  pairs: MultiValue<OptionType>;
  useLatestData: boolean;
}

interface PropsSelectBulkSimPairs {
  onSelect: (values: MultiValue<OptionType>) => void;
}

const SelectBulkSimPairsBody = ({ onSelect }: PropsSelectBulkSimPairs) => {
  return (
    <div>
      {BULK_SIM_PAIR_PRESETS.map((preset) => {
        return (
          <div>
            <Tooltip label={preset.pairs.map((pair) => pair.label).join(", ")}>
              <Button
                variant={BUTTON_VARIANTS.nofill}
                onClick={() => {
                  onSelect(preset.pairs);
                }}
              >
                {preset.label} ({preset.pairs.length} symbols)
              </Button>
            </Tooltip>
          </div>
        );
      })}
    </div>
  );
};

export const BacktestOnManyPairs = () => {
  const { backtestId } = usePathParams<PathParams>();
  const { runBacktestOnManyPairsModal } = useBacktestContext();

  const binanceTickersQuery = useBinanceTickersQuery();
  const backtestQuery = useBacktestById(Number(backtestId));
  const presetsPopover = useDisclosure();

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
      fetch_latest_data: formValues.useLatestData,
    });

    if (res.status === 200) {
      toast({
        title: "Started mass backtest",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      runBacktestOnManyPairsModal.onClose();
    }
  };

  return (
    <ChakraModal
      {...runBacktestOnManyPairsModal}
      title="Run bulk sim"
      modalContentStyle={{ maxWidth: "60%" }}
    >
      <Formik initialValues={getFormInitialValues()} onSubmit={onSubmit}>
        {({ setFieldValue, values }) => {
          return (
            <Form>
              <FormControl>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    justifyContent: "space-between",
                  }}
                >
                  <FormLabel fontSize={"x-large"}>Select pairs</FormLabel>

                  <ChakraPopover
                    {...presetsPopover}
                    setOpen={presetsPopover.onOpen}
                    body={
                      <SelectBulkSimPairsBody
                        onSelect={(values) =>
                          setFieldValue(formKeys.pairs, values)
                        }
                      />
                    }
                    headerText="Select pairs from a preset"
                  >
                    <Button variant={BUTTON_VARIANTS.nofill}>Presets</Button>
                  </ChakraPopover>
                </div>
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
              </FormControl>

              <div style={{ marginTop: "16px" }}>
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
