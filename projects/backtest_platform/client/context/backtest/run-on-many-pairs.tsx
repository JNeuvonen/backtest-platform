import React from "react";
import { ChakraModal } from "../../components/chakra/modal";
import { useBacktestContext } from ".";
import { Field, Form, Formik } from "formik";
import {
  Button,
  FormControl,
  FormLabel,
  Heading,
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
  useNyseSymbolList,
  useTopCoinsByVol,
} from "../../clients/queries/queries";
import { usePathParams } from "../../hooks/usePathParams";
import { postMassBacktest } from "../../clients/requests";
import { WithLabel } from "../../components/form/WithLabel";
import { BUTTON_VARIANTS } from "../../theme";
import { ChakraPopover } from "../../components/chakra/popover";
import {
  BULK_SIM_NYSE_SYMBOL_PRESETS,
  BULK_SIM_PAIR_PRESETS,
} from "../../utils/hardcodedpresets";

const formKeys = {
  pairs: "pairs",
  stockMarketSymbols: "stockMarketSymbols",
  useLatestData: "useLatestData",
};

const getFormInitialValues = () => {
  return {
    pairs: [] as MultiValue<OptionType>,
    useLatestData: true,
    stockMarketSymbols: [] as MultiValue<OptionType>,
  };
};

interface PathParams {
  datasetName: string;
  backtestId: number;
}

export interface MassBacktest {
  pairs: MultiValue<OptionType>;
  stockMarketSymbols: MultiValue<OptionType>;
  useLatestData: boolean;
}

interface PropsSelectBulkSimPairs {
  onSelect: (values: MultiValue<OptionType>) => void;
}

export const SelectBulkSimPairsBody = ({
  onSelect,
}: PropsSelectBulkSimPairs) => {
  const top20CoinsByVol = useTopCoinsByVol(25);
  const top50CoinsByVol = useTopCoinsByVol(50);
  const top100CoinsByVol = useTopCoinsByVol(100);
  const top200CoinsByVol = useTopCoinsByVol(200);

  const renderTopCoinsByVol = () => {
    if (
      !top20CoinsByVol.data ||
      !top50CoinsByVol.data ||
      !top100CoinsByVol.data ||
      !top200CoinsByVol.data
    ) {
      return <Spinner />;
    }

    return (
      <div style={{ marginTop: "16px" }}>
        <Heading size={"md"}>Real-time top list by vol.</Heading>
        <div style={{ marginTop: "12px" }}>
          <Tooltip label={top20CoinsByVol.data.map((pair) => pair).join(", ")}>
            <Button
              variant={BUTTON_VARIANTS.nofill}
              onClick={() => {
                onSelect(
                  top20CoinsByVol.data.map((item) => {
                    return {
                      value: item,
                      label: item,
                    };
                  })
                );
              }}
            >
              Top 20
            </Button>
          </Tooltip>
        </div>
        <div style={{ marginTop: "2px" }}>
          <Tooltip label={top50CoinsByVol.data.map((pair) => pair).join(", ")}>
            <Button
              variant={BUTTON_VARIANTS.nofill}
              onClick={() => {
                onSelect(
                  top50CoinsByVol.data.map((item) => {
                    return {
                      value: item,
                      label: item,
                    };
                  })
                );
              }}
            >
              Top 50
            </Button>
          </Tooltip>
        </div>

        <div style={{ marginTop: "2px" }}>
          <Tooltip label={top100CoinsByVol.data.map((pair) => pair).join(", ")}>
            <Button
              variant={BUTTON_VARIANTS.nofill}
              onClick={() => {
                onSelect(
                  top100CoinsByVol.data.map((item) => {
                    return {
                      value: item,
                      label: item,
                    };
                  })
                );
              }}
            >
              Top 100
            </Button>
          </Tooltip>
        </div>
        <div style={{ marginTop: "2px" }}>
          <Tooltip label={top200CoinsByVol.data.map((pair) => pair).join(", ")}>
            <Button
              variant={BUTTON_VARIANTS.nofill}
              onClick={() => {
                onSelect(
                  top200CoinsByVol.data.map((item) => {
                    return {
                      value: item,
                      label: item,
                    };
                  })
                );
              }}
            >
              Top 200
            </Button>
          </Tooltip>
        </div>
      </div>
    );
  };

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
      {renderTopCoinsByVol()}
    </div>
  );
};

export const SelectBulkSimNyseSymbolsBody = ({
  onSelect,
}: PropsSelectBulkSimPairs) => {
  return (
    <div>
      {BULK_SIM_NYSE_SYMBOL_PRESETS.map((preset) => {
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
  const nyseTickersQuery = useNyseSymbolList();
  const backtestQuery = useBacktestById(Number(backtestId));
  const cryptoPresetsPopover = useDisclosure();
  const stockMarketPresetsPopover = useDisclosure();

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
      crypto_symbols: symbols,
      stock_market_symbols: formValues.stockMarketSymbols.map(
        (item) => item.value
      ),
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
                  <FormLabel fontSize={"x-large"}>Cryptocurrency</FormLabel>
                  <ChakraPopover
                    {...cryptoPresetsPopover}
                    setOpen={cryptoPresetsPopover.onOpen}
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
              <FormControl>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    justifyContent: "space-between",
                    marginTop: "16px",
                  }}
                >
                  <FormLabel fontSize={"x-large"}>Stock market</FormLabel>
                  <ChakraPopover
                    {...stockMarketPresetsPopover}
                    setOpen={stockMarketPresetsPopover.onOpen}
                    body={
                      <SelectBulkSimNyseSymbolsBody
                        onSelect={(values) =>
                          setFieldValue(formKeys.stockMarketSymbols, values)
                        }
                      />
                    }
                    headerText="Select pairs from a preset"
                  >
                    <Button variant={BUTTON_VARIANTS.nofill}>Presets</Button>
                  </ChakraPopover>
                </div>
                <Field
                  name={formKeys.stockMarketSymbols}
                  as={SelectWithTextFilter}
                  options={nyseTickersQuery.data?.map((item) => {
                    return {
                      label: item,
                      value: item,
                    };
                  })}
                  onChange={(selectedOptions: MultiValue<OptionType>) =>
                    setFieldValue(formKeys.stockMarketSymbols, selectedOptions)
                  }
                  isMulti={true}
                  closeMenuOnSelect={false}
                  value={values.stockMarketSymbols}
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
