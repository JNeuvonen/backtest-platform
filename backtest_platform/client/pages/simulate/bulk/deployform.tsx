import React, { useRef } from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useBacktestById,
  useMassbacktestSymbols,
  useMassbacktestTransformations,
} from "../../../clients/queries/queries";
import { Field, Form, Formik, FormikProps } from "formik";
import { WithLabel } from "../../../components/form/WithLabel";
import { ChakraInput } from "../../../components/chakra/input";
import { NumberInput, NumberInputField, useToast } from "@chakra-ui/react";
import { ChakraNumberStepper } from "../../../components/ChakraNumberStepper";
import { FormSubmitBar } from "../../../components/form/FormSubmitBar";
import {
  TradeQuantityPrecision,
  getIntervalLengthInMs,
  getSymbolsWithPrecision,
} from "../../../utils/binance";
import { deployPairtradeSystem } from "../../../clients/requests";
import { useAppContext } from "../../../context/app";

interface PathParams {
  massPairTradeBacktestId: number;
}

const formKeys = {
  name: "name",
  maxSimultaneousPositions: "maxSimultaneousPositions",
  numReqKlines: "numReqKlines",
  maximumKlinesHoldTime: "maximumKlinesHoldTime",
  maxLeverageRatio: "maxLeverageRatio",
  loanFailRetryCooldown: "loanFailRetryCooldown",
};

export interface LongShortDeployForm {
  name: string;
  candle_interval: string;
  buy_cond: string;
  sell_cond: string;
  exit_cond: string;
  num_req_klines: number;
  max_simultaneous_positions: number;
  kline_size_ms: number;
  klines_until_close: number;
  max_leverage_ratio: number;
  take_profit_threshold_perc: number;
  stop_loss_threshold_perc: number;
  use_time_based_close: boolean;
  use_profit_based_close: boolean;
  use_stop_loss_based_close: boolean;
  use_taker_order: boolean;
  asset_universe: TradeQuantityPrecision[];
  data_transformations: object[];
}

const getFormInitialValues = () => {
  return {
    [formKeys.name]: "",
    [formKeys.maxSimultaneousPositions]: 65,
    [formKeys.numReqKlines]: 100,
    [formKeys.maxLeverageRatio]: 1.2,
    [formKeys.loanFailRetryCooldown]: 3600000,
  };
};

interface Props {
  onSuccessCallback: () => void;
}

export const DeployLongShortStrategyForm = ({ onSuccessCallback }: Props) => {
  const { massPairTradeBacktestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(massPairTradeBacktestId));
  const symbolsQuery = useMassbacktestSymbols(Number(massPairTradeBacktestId));
  const { getPredServAPIKey } = useAppContext();
  const transformationsQuery = useMassbacktestTransformations(
    Number(massPairTradeBacktestId)
  );
  const toast = useToast();

  const formikRef = useRef<FormikProps<any>>(null);

  const onSubmit = async (values) => {
    if (
      !backtestQuery.data?.data ||
      !symbolsQuery.data ||
      !transformationsQuery.data
    ) {
      return;
    }

    const backtest = backtestQuery.data.data;

    const assetUniverse = await getSymbolsWithPrecision(
      symbolsQuery.data.map((item) => item.symbol)
    );

    const body = {
      name: values[formKeys.name],
      candle_interval: backtest.candle_interval,
      buy_cond: backtest.long_short_buy_cond,
      sell_cond: backtest.long_short_sell_cond,
      exit_cond: backtest.long_short_exit_cond,
      num_req_klines: values[formKeys.numReqKlines],
      max_simultaneous_positions: values[formKeys.maxSimultaneousPositions],
      kline_size_ms: getIntervalLengthInMs(backtest.candle_interval),
      klines_until_close: backtest.klines_until_close,
      max_leverage_ratio: values[formKeys.maxLeverageRatio],
      loan_retry_wait_time_ms: values[formKeys.loanFailRetryCooldown],
      take_profit_threshold_perc: backtest.take_profit_threshold_perc,
      stop_loss_threshold_perc: backtest.stop_loss_threshold_perc,
      use_time_based_close: backtest.use_time_based_close,
      use_profit_based_close: backtest.use_profit_based_close,
      use_stop_loss_based_close: backtest.use_stop_loss_based_close,
      use_taker_order: true,
      asset_universe: assetUniverse,
      data_transformations: transformationsQuery.data.map((item) => {
        return {
          id: item.id,
          created_at: new Date(item.created_at),
          updated_at: new Date(item.updated_at),
          transformation_code: item.transformation_code,
        };
      }),
    };

    const res = await deployPairtradeSystem(getPredServAPIKey(), body);

    if (res.status === 200) {
      toast({
        title: "Deployed long/short system",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      onSuccessCallback();
    }
  };

  return (
    <div>
      <Formik
        onSubmit={(values) => {
          onSubmit(values);
        }}
        initialValues={getFormInitialValues()}
        innerRef={formikRef}
        enableReinitialize
      >
        {() => (
          <Form>
            <div>
              <Field name={formKeys.name}>
                {({ form, field }) => {
                  return (
                    <WithLabel>
                      <ChakraInput
                        label="Strategy name"
                        onChange={(value: string) =>
                          form.setFieldValue(formKeys.name, value)
                        }
                        value={field.value}
                      />
                    </WithLabel>
                  );
                }}
              </Field>
            </div>

            <div style={{ display: "flex", gap: "16px", marginTop: "16px" }}>
              <div>
                <Field name={formKeys.maxSimultaneousPositions}>
                  {({ field, form }) => {
                    return (
                      <WithLabel
                        label={"Max simultaneous pos."}
                        containerStyles={{
                          maxWidth: "200px",
                        }}
                      >
                        <NumberInput
                          step={1}
                          min={0}
                          value={field.value}
                          onChange={(valueString) =>
                            form.setFieldValue(
                              formKeys.maxSimultaneousPositions,
                              parseInt(valueString)
                            )
                          }
                        >
                          <NumberInputField />
                          <ChakraNumberStepper />
                        </NumberInput>
                      </WithLabel>
                    );
                  }}
                </Field>
              </div>

              <div>
                <Field name={formKeys.numReqKlines}>
                  {({ field, form }) => {
                    return (
                      <WithLabel
                        label={"Num of required candles"}
                        containerStyles={{
                          maxWidth: "200px",
                        }}
                      >
                        <NumberInput
                          step={10000}
                          min={0}
                          value={field.value}
                          onChange={(valueString) =>
                            form.setFieldValue(
                              formKeys.numReqKlines,
                              parseInt(valueString)
                            )
                          }
                        >
                          <NumberInputField />
                          <ChakraNumberStepper />
                        </NumberInput>
                      </WithLabel>
                    );
                  }}
                </Field>
              </div>
              <div>
                <Field name={formKeys.maxLeverageRatio}>
                  {({ field, form }) => {
                    return (
                      <WithLabel
                        label={"Max leverage ratio"}
                        containerStyles={{
                          maxWidth: "200px",
                        }}
                      >
                        <NumberInput
                          step={5}
                          min={0}
                          value={field.value}
                          onChange={(valueString) =>
                            form.setFieldValue(
                              formKeys.maximumKlinesHoldTime,
                              parseInt(valueString)
                            )
                          }
                        >
                          <NumberInputField />
                          <ChakraNumberStepper />
                        </NumberInput>
                      </WithLabel>
                    );
                  }}
                </Field>
              </div>
              <div>
                <Field name={formKeys.loanFailRetryCooldown}>
                  {({ field, form }) => {
                    return (
                      <WithLabel
                        label={`Loan fail retry cooldown (${
                          field.value / 1000 / 60
                        } mins)`}
                        containerStyles={{
                          maxWidth: "350px",
                        }}
                      >
                        <NumberInput
                          step={300000}
                          min={0}
                          value={field.value}
                          onChange={(valueString) =>
                            form.setFieldValue(
                              formKeys.loanFailRetryCooldown,
                              parseInt(valueString)
                            )
                          }
                        >
                          <NumberInputField />
                          <ChakraNumberStepper />
                        </NumberInput>
                      </WithLabel>
                    );
                  }}
                </Field>
              </div>
            </div>

            <FormSubmitBar style={{ marginTop: "16px" }} />
          </Form>
        )}
      </Formik>
    </div>
  );
};
