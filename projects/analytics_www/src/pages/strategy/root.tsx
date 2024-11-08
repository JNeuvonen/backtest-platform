import {
  Badge,
  Button,
  Heading,
  NumberInput,
  NumberInputField,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
  Switch,
  Text,
  useDisclosure,
} from "@chakra-ui/react";
import {
  getDiffToPresentFormatted,
  getNumberDisplayColor,
  getStrategyGroupTradeInfo,
  roundNumberFloor,
} from "src/common_js";
import { useNavigate } from "react-router-dom";
import {
  ChakraAccordion,
  ChakraCard,
  ChakraDrawer,
  ChakraInput,
  ChakraNumberStepper,
} from "src/components/chakra";
import { usePathParams } from "src/hooks";
import {
  useBalanceSnapshotsQuery,
  useBinanceSpotPriceInfo,
  useStrategyGroupQuery,
} from "src/http/queries";
import {
  BUTTON_VARIANTS,
  COLOR_BG_TERTIARY,
  COLOR_CONTENT_PRIMARY,
} from "src/theme";
import {
  getStrategyCompletedTradesPath,
  getStrategySymbolsPath,
} from "src/utils";
import { ConfirmModal } from "src/components/ConfirmModal";
import {
  disableAndCloseStratGroup,
  enableStratGroupRequest,
  updateStratGroupRiskParams,
} from "src/http";
import { toast } from "react-toastify";
import { ReadOnlyEditor } from "src/components/ReadOnlyEditor";
import { WithLabel } from "src/components/WithLabel";
import { OpenTradesTable } from "src/components/OpenTradesTable";
import { StrategyGroup } from "src/common_js";
import { Field, Form, Formik } from "formik";
import { FormSubmitBar } from "src/components";
import { useAppContext } from "src/context";

const EDIT_STRAT_GROUP_FORM_KEYS = {
  numSymbolsForAutoAdaptive: "numSymbolsForAutoAdaptive",
  adaptiveGroupsRefreshPeriod: "adaptiveGroupsRefreshPeriod",
  allocationPerSymbol: "allocationPerSymbol",
};

interface FormValues {
  numSymbolsForAutoAdaptive: number;
  adaptiveGroupsRefreshPeriod: number;
  allocationPerSymbol: number;
}

export const EditStratGroupSettingsDrawer = ({
  strategyGroup,
  onSuccessCallback,
}: {
  strategyGroup: StrategyGroup;
  onSuccessCallback: () => void;
}) => {
  const submit = async (values: FormValues) => {
    const res = await updateStratGroupRiskParams(strategyGroup.id, {
      num_symbols_for_auto_adaptive: values.numSymbolsForAutoAdaptive,
      num_days_for_group_recalc: values.adaptiveGroupsRefreshPeriod,
      allocated_size_perc: values.allocationPerSymbol,
    });

    if (res.success) {
      toast.success("Updated strategy group", { theme: "dark" });
      onSuccessCallback();
    } else {
      toast.error("Failed to update strategy group", { theme: "dark" });
    }
  };

  return (
    <div>
      <Formik
        initialValues={{
          numSymbolsForAutoAdaptive:
            strategyGroup.num_symbols_for_auto_adaptive,
          adaptiveGroupsRefreshPeriod: strategyGroup.num_days_for_group_recalc,
          allocationPerSymbol: strategyGroup.allocated_size_perc,
        }}
        onSubmit={(values) => {
          submit(values);
        }}
      >
        <Form>
          <div>
            <div style={{ marginTop: "16px" }}>
              <Field
                name={EDIT_STRAT_GROUP_FORM_KEYS.allocationPerSymbol}
                component={({ field, form }) => {
                  return (
                    <WithLabel
                      label={"Allocation per symbol"}
                      containerStyles={{
                        maxWidth: "200px",
                      }}
                    >
                      <NumberInput
                        step={0.25}
                        min={0}
                        max={100}
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

            <div style={{ marginTop: "16px" }}>
              <Field
                name={EDIT_STRAT_GROUP_FORM_KEYS.adaptiveGroupsRefreshPeriod}
                component={({ field, form }) => {
                  return (
                    <WithLabel
                      label={"Number of days to update group symbols"}
                      containerStyles={{
                        maxWidth: "200px",
                      }}
                    >
                      <NumberInput
                        step={1}
                        min={0}
                        max={365}
                        value={field.value}
                        onChange={(value) => {
                          form.setFieldValue(field.name, parseInt(value));
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

            <div style={{ marginTop: "16px" }}>
              <Field
                name={EDIT_STRAT_GROUP_FORM_KEYS.numSymbolsForAutoAdaptive}
                component={({ field, form }) => {
                  return (
                    <WithLabel
                      label={"Number of symbols in group"}
                      containerStyles={{
                        maxWidth: "200px",
                      }}
                    >
                      <NumberInput
                        step={1}
                        min={0}
                        max={365}
                        value={field.value}
                        onChange={(value) => {
                          form.setFieldValue(field.name, parseInt(value));
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
          </div>
        </Form>
      </Formik>
    </div>
  );
};

export const ViewStrategyGroupSettings = ({ strategyGroup }) => {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        flexWrap: "wrap",
        gap: "16px",
        marginTop: "16px",
      }}
    >
      <div>
        <WithLabel label={"Is enabled"}>
          <Switch isChecked={!strategyGroup.is_disabled} isDisabled={true} />
        </WithLabel>
      </div>
      <div>
        <WithLabel label={"Is auto adaptive group"}>
          <Switch
            isChecked={strategyGroup.is_auto_adaptive_group}
            isDisabled={true}
          />
        </WithLabel>
      </div>

      {strategyGroup.is_auto_adaptive_group && (
        <>
          <div>
            <WithLabel
              label={"Num symbols for auto adaptive"}
              containerStyles={{
                maxWidth: "200px",
              }}
            >
              <NumberInput
                isDisabled={true}
                value={strategyGroup.num_symbols_for_auto_adaptive}
              >
                <NumberInputField />
                <ChakraNumberStepper />
              </NumberInput>
            </WithLabel>
          </div>
          <div>
            <WithLabel
              label={"Num days for group symbols refresh"}
              containerStyles={{
                maxWidth: "200px",
              }}
            >
              <NumberInput
                isDisabled={true}
                value={strategyGroup.num_days_for_group_recalc}
              >
                <NumberInputField />
                <ChakraNumberStepper />
              </NumberInput>
            </WithLabel>
          </div>
        </>
      )}
      <div>
        <WithLabel label={"Calc stops on pred server"}>
          <Switch
            isChecked={strategyGroup.should_calc_stops_on_pred_serv}
            isDisabled={true}
          />
        </WithLabel>
      </div>
      <div>
        <WithLabel label={"Use stop loss"}>
          <Switch
            isChecked={strategyGroup.use_stop_loss_based_close}
            isDisabled={true}
          />
        </WithLabel>
      </div>

      {strategyGroup.use_stop_loss_based_close && (
        <div>
          <WithLabel
            label={"Stop loss threshold (%)"}
            containerStyles={{
              maxWidth: "200px",
            }}
          >
            <NumberInput
              isDisabled={true}
              value={strategyGroup.stop_loss_threshold_perc}
            >
              <NumberInputField />
              <ChakraNumberStepper />
            </NumberInput>
          </WithLabel>
        </div>
      )}

      <div>
        <WithLabel label={"Use profit based close"}>
          <Switch
            isChecked={strategyGroup.use_profit_based_close}
            isDisabled={true}
          />
        </WithLabel>
      </div>

      {strategyGroup.use_profit_based_close && (
        <div>
          <WithLabel
            label={"Stop loss threshold (%)"}
            containerStyles={{
              maxWidth: "200px",
            }}
          >
            <NumberInput
              isDisabled={true}
              value={strategyGroup.take_profit_threshold_perc}
            >
              <NumberInputField />
              <ChakraNumberStepper />
            </NumberInput>
          </WithLabel>
        </div>
      )}

      <div>
        <WithLabel label={"Use time based close"}>
          <Switch
            isChecked={strategyGroup.use_time_based_close}
            isDisabled={true}
          />
        </WithLabel>
      </div>

      {strategyGroup.use_time_based_close && (
        <div>
          <WithLabel
            label={"Maximum candles hold time"}
            containerStyles={{
              maxWidth: "200px",
            }}
          >
            <NumberInput
              isDisabled={true}
              value={strategyGroup.maximum_klines_hold_time}
            >
              <NumberInputField />
              <ChakraNumberStepper />
            </NumberInput>
          </WithLabel>
        </div>
      )}
      <div>
        <ChakraInput
          label={"Candle interval"}
          value={strategyGroup.candle_interval}
          disabled={true}
        />
      </div>
      <div>
        <WithLabel
          label={"Num req klines"}
          containerStyles={{
            maxWidth: "200px",
          }}
        >
          <NumberInput isDisabled={true} value={strategyGroup.num_req_klines}>
            <NumberInputField />
            <ChakraNumberStepper />
          </NumberInput>
        </WithLabel>
      </div>
      <div>
        <WithLabel label={"Is leverage allowed"}>
          <Switch
            isChecked={strategyGroup.is_leverage_allowed}
            isDisabled={true}
          />
        </WithLabel>
      </div>
      <div>
        <WithLabel label={"Use taker order"}>
          <Switch isChecked={strategyGroup.use_taker_order} isDisabled={true} />
        </WithLabel>
      </div>
    </div>
  );
};

export const StrategyPage = () => {
  const { strategyName } = usePathParams<{ strategyName: string }>();
  const { isMobileLayout } = useAppContext();
  const strategyGroupQuery = useStrategyGroupQuery(strategyName);
  const binancePriceQuery = useBinanceSpotPriceInfo();
  const balanceSnapShots = useBalanceSnapshotsQuery();
  const navigate = useNavigate();
  const disableAndCloseConfirmModal = useDisclosure();
  const enableConfirmModal = useDisclosure();
  const editGroupSettingsDrawer = useDisclosure();

  if (
    strategyGroupQuery.isLoading ||
    !strategyGroupQuery.data ||
    !binancePriceQuery.data ||
    !balanceSnapShots.data ||
    balanceSnapShots.isLoading
  ) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const disableAndClosePositions = async () => {
    const res = await disableAndCloseStratGroup(
      strategyGroupQuery.data.strategy_group.id,
    );

    if (res.success) {
      toast.success("Disabled strategy", { theme: "dark" });
      strategyGroupQuery.refetch();
    } else {
      toast.error("Failed to disable strategy", { theme: "dark" });
    }
  };

  const enableStrategyGroup = async () => {
    const res = await enableStratGroupRequest(
      strategyGroupQuery.data.strategy_group.id,
    );

    if (res.success) {
      toast.success("Enabled strategy", { theme: "dark" });
      strategyGroupQuery.refetch();
    } else {
      toast.error("Failed to disable strategy", { theme: "dark" });
    }
  };

  const renderStratEnabledBadge = () => {
    if (strategyGroupQuery.data.strategy_group.is_disabled) {
      return <Badge colorScheme="red">Disabled</Badge>;
    }
    return <Badge colorScheme="green">Enabled</Badge>;
  };

  const tradeInfoDict = getStrategyGroupTradeInfo(
    strategyGroupQuery.data.strategy_group,
    strategyGroupQuery.data.strategies,
    strategyGroupQuery.data.trades,
    binancePriceQuery.data,
  );

  const lastBalanceSnapshot =
    balanceSnapShots.data[balanceSnapShots.data.length - 1];

  return (
    <div>
      <ConfirmModal
        {...disableAndCloseConfirmModal}
        onConfirm={disableAndClosePositions}
        title={"Disable strategy"}
        message={
          "Are you sure you want to disable this strategy and close all positions?"
        }
      />
      <ConfirmModal
        {...enableConfirmModal}
        onConfirm={enableStrategyGroup}
        title={"Disable strategy"}
        message={"Are you sure you want to enable this strategy?"}
      />

      <ChakraDrawer
        {...editGroupSettingsDrawer}
        title={"Edit group settings"}
        drawerContentStyles={{
          maxWidth: isMobileLayout ? "90%" : "40%",
          minWidth: "300px",
        }}
      >
        <EditStratGroupSettingsDrawer
          strategyGroup={strategyGroupQuery.data.strategy_group}
          onSuccessCallback={() => {
            strategyGroupQuery.refetch();
            editGroupSettingsDrawer.onClose();
          }}
        />
      </ChakraDrawer>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: "16px",
          flexWrap: "wrap",
        }}
      >
        <div>
          <Heading size={"lg"}>
            {strategyGroupQuery.data.strategy_group.name}
          </Heading>
          <Text fontSize={"13px"}>
            Went live:{" "}
            {getDiffToPresentFormatted(
              new Date(strategyGroupQuery.data.strategy_group.created_at),
            )}{" "}
            ago
          </Text>
        </div>

        <div>{renderStratEnabledBadge()}</div>
      </div>
      <div style={{ marginTop: "8px" }}>
        <ChakraCard>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "16px",
              flexWrap: "wrap",
            }}
          >
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Inception date</StatLabel>
                <StatNumber>
                  {new Date(
                    strategyGroupQuery.data.strategy_group.created_at,
                  ).toLocaleString("default", {
                    year: "numeric",
                    month: "short",
                    day: "numeric",
                  })}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Universe size</StatLabel>
                <StatNumber>{tradeInfoDict.numStrategies}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Trades</StatLabel>
                <StatNumber>{tradeInfoDict.totalTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Open trades</StatLabel>
                <StatNumber>{tradeInfoDict.openTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Closed trades</StatLabel>
                <StatNumber>{tradeInfoDict.closedTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Realized profit</StatLabel>
                <StatNumber
                  color={getNumberDisplayColor(
                    tradeInfoDict.cumulativeNetResult,
                    COLOR_CONTENT_PRIMARY,
                  )}
                >
                  ${roundNumberFloor(tradeInfoDict.cumulativeNetResult, 2)}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Unrealized profit</StatLabel>
                <StatNumber
                  color={getNumberDisplayColor(
                    tradeInfoDict.cumulativeUnrealizedProfit,
                    COLOR_CONTENT_PRIMARY,
                  )}
                >
                  $
                  {roundNumberFloor(
                    tradeInfoDict.cumulativeUnrealizedProfit,
                    2,
                  )}
                </StatNumber>
              </Stat>
            </div>

            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Mean allocation</StatLabel>
                <StatNumber>
                  {roundNumberFloor(tradeInfoDict.meanAllocation, 2)}%
                </StatNumber>
              </Stat>
            </div>

            {lastBalanceSnapshot && (
              <div>
                <Stat color={COLOR_CONTENT_PRIMARY}>
                  <StatLabel>Value at risk</StatLabel>
                  <StatNumber>
                    {roundNumberFloor(
                      (tradeInfoDict.positionSize / lastBalanceSnapshot.value) *
                        100,
                      2,
                    )}
                    %
                  </StatNumber>
                </Stat>
              </div>
            )}

            {strategyGroupQuery.data.strategy_group.is_auto_adaptive_group && (
              <div>
                <Stat color={COLOR_CONTENT_PRIMARY}>
                  <StatLabel>Group symbols last updated</StatLabel>
                  <StatNumber>
                    {getDiffToPresentFormatted(
                      new Date(
                        strategyGroupQuery.data.strategy_group.last_adaptive_group_recalc,
                      ),
                    )}
                    {" ago"}
                  </StatNumber>
                </Stat>
              </div>
            )}
          </div>
        </ChakraCard>
      </div>

      <div
        style={{
          marginTop: "16px",
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <Button
          variant={BUTTON_VARIANTS.dangerNoFill}
          onClick={() => disableAndCloseConfirmModal.onOpen()}
        >
          Disable and close positions
        </Button>
        {strategyGroupQuery.data.strategy_group.is_disabled && (
          <div>
            <Button
              variant={BUTTON_VARIANTS.dangerNoFill}
              onClick={() => enableConfirmModal.onOpen()}
            >
              Enable
            </Button>
          </div>
        )}
        <Button
          variant={BUTTON_VARIANTS.nofill}
          onClick={() => navigate(getStrategySymbolsPath(strategyName))}
        >
          Strategy symbols
        </Button>
        <Button
          variant={BUTTON_VARIANTS.nofill}
          onClick={() => navigate(getStrategyCompletedTradesPath(strategyName))}
        >
          View trades
        </Button>

        <Button
          variant={BUTTON_VARIANTS.nofill}
          onClick={editGroupSettingsDrawer.onOpen}
        >
          Edit group settings
        </Button>
      </div>

      <div>
        <Heading size={"md"}>Enter trade criteria</Heading>
        <ReadOnlyEditor
          value={strategyGroupQuery.data.strategy_group.enter_trade_code}
          style={{ marginTop: "8px" }}
          height={150}
        />
      </div>
      <div style={{ marginTop: "16px" }}>
        <Heading size={"md"}>Exit trade criteria</Heading>
        <ReadOnlyEditor
          value={strategyGroupQuery.data.strategy_group.exit_trade_code}
          style={{ marginTop: "8px" }}
          height={150}
        />
      </div>
      <ChakraAccordion
        heading="Strategy parameters"
        containerStyles={{ marginTop: "10px", border: COLOR_BG_TERTIARY }}
      >
        <ViewStrategyGroupSettings
          strategyGroup={strategyGroupQuery.data.strategy_group}
        />
      </ChakraAccordion>

      <div style={{ marginTop: "16px" }}>
        <Heading size={"md"}>Open positions</Heading>
        <OpenTradesTable
          trades={strategyGroupQuery.data.trades.filter(
            (item) => item.close_price === null,
          )}
        />
      </div>
    </div>
  );
};
