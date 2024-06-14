import {
  Button,
  FormControl,
  FormLabel,
  Select,
  Spinner,
  Switch,
  UseDisclosureReturn,
  useDisclosure,
} from "@chakra-ui/react";
import React from "react";
import { ChakraModal } from "./chakra/modal";
import { OptionType, SelectWithTextFilter } from "./SelectFilter";
import { MultiValue } from "react-select";
import { FormSubmitBar } from "./form/FormSubmitBar";
import {
  useBinanceTickersQuery,
  useNyseSymbolList,
} from "../clients/queries/queries";
import { Field, Form, Formik } from "formik";
import { ChakraPopover } from "./chakra/popover";
import {
  SelectBulkSimNyseSymbolsBody,
  SelectBulkSimPairsBody,
} from "../context/backtest/run-on-many-pairs";
import { binanceTickSelectOptions } from "../pages/Datasets";
import { BUTTON_VARIANTS } from "../theme";
import { GET_KLINE_OPTIONS } from "../utils/constants";

interface Props {
  modalControls: UseDisclosureReturn;
  onSelectStockMarketSymbols: (selectedItems: MultiValue<OptionType>) => void;
  onSelectCryptoSymbols: (selectedItems: MultiValue<OptionType>) => void;
  onSelectCandleInterval: (candleInterval: string) => void;
}

const formKeys = {
  cryptoSymbols: "cryptoSymbols",
  stockSymbols: "stockSymbols",
  candleInterval: "candleInterval",
};

const getFormInitialValues = () => {
  return {
    cryptoSymbols: [],
    stockSymbols: [],
    candleInterval: "1d",
  };
};

export const SelectUniverseModal = ({
  modalControls,
  onSelectStockMarketSymbols,
  onSelectCryptoSymbols,
  onSelectCandleInterval,
}: Props) => {
  const binanceTickersQuery = useBinanceTickersQuery();
  const nyseTickersQuery = useNyseSymbolList();
  const cryptoPresetsPopover = useDisclosure();
  const stockMarketPresetsPopover = useDisclosure();

  if (!binanceTickersQuery.data || !nyseTickersQuery.data) {
    return (
      <ChakraModal {...modalControls} title={"Select universe"}>
        <Spinner />
      </ChakraModal>
    );
  }
  return (
    <ChakraModal {...modalControls} title="Select universe">
      <Formik
        initialValues={getFormInitialValues()}
        onSubmit={(values) => {
          onSelectCryptoSymbols(values.cryptoSymbols);
          onSelectStockMarketSymbols(values.stockSymbols);
          onSelectCandleInterval(values.candleInterval);
          modalControls.onClose();
        }}
      >
        {({ setFieldValue, values }) => {
          return (
            <Form>
              <div>
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
                          setFieldValue(formKeys.cryptoSymbols, values)
                        }
                      />
                    }
                    headerText="Select pairs from a preset"
                  >
                    <Button variant={BUTTON_VARIANTS.nofill}>Presets</Button>
                  </ChakraPopover>
                </div>
                <div style={{ marginTop: "8px" }}>
                  <Field
                    name={formKeys.cryptoSymbols}
                    as={SelectWithTextFilter}
                    options={binanceTickSelectOptions(
                      binanceTickersQuery.data.res.pairs
                    )}
                    onChange={(selectedOptions: MultiValue<OptionType>) =>
                      setFieldValue(formKeys.cryptoSymbols, selectedOptions)
                    }
                    isMulti={true}
                    closeMenuOnSelect={false}
                    value={values.cryptoSymbols}
                  />
                </div>
              </div>
              <div style={{ marginTop: "16px" }}>
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
                          setFieldValue(formKeys.stockSymbols, values)
                        }
                      />
                    }
                    headerText="Select pairs from a preset"
                  >
                    <Button variant={BUTTON_VARIANTS.nofill}>Presets</Button>
                  </ChakraPopover>
                </div>
                <div style={{ marginTop: "8px" }}>
                  <Field
                    name={formKeys.stockSymbols}
                    as={SelectWithTextFilter}
                    options={nyseTickersQuery.data?.map((item) => {
                      return {
                        label: item,
                        value: item,
                      };
                    })}
                    onChange={(selectedOptions: MultiValue<OptionType>) =>
                      setFieldValue(formKeys.stockSymbols, selectedOptions)
                    }
                    isMulti={true}
                    closeMenuOnSelect={false}
                    value={values.stockSymbols}
                  />
                </div>
              </div>

              <FormControl marginTop={"8px"}>
                <FormLabel fontSize={"x-large"}>Candle interval</FormLabel>
                <Field
                  name={formKeys.candleInterval}
                  as={Select}
                  value={values.candleInterval}
                >
                  {GET_KLINE_OPTIONS().map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </Field>
              </FormControl>

              <div style={{ marginTop: "32px" }}>
                <FormSubmitBar
                  submitText="Ok"
                  cancelCallback={modalControls.onClose}
                />
              </div>
            </Form>
          );
        }}
      </Formik>
    </ChakraModal>
  );
};
