import React from "react";
import { ChakraCard } from "../../../../components/chakra/Card";
import { Heading } from "@chakra-ui/react";
import { COLOR_CONTENT_PRIMARY } from "../../../../utils/colors";
import {
  DatasetModel,
  FetchBacktestByIdRes,
} from "../../../../clients/queries/response-types";
import { UseQueryResult } from "@tanstack/react-query";

interface Props {
  backtestQuery: UseQueryResult<FetchBacktestByIdRes | null, unknown>;
  modelQuery?: UseQueryResult<DatasetModel | null, unknown>;
}
export const TradingCriteriaCard = (props: Props) => {
  const { backtestQuery, modelQuery } = props;
  if (backtestQuery.data === undefined || backtestQuery.data === null) {
    return;
  }

  const getStrategyType = () => {
    if (backtestQuery.data === undefined || backtestQuery.data === null) {
      return;
    }

    if (backtestQuery.data?.data.is_ml_based_strategy) {
      return "ML based";
    } else {
      console.log(backtestQuery.data.data.is_short_selling_strategy);
      return backtestQuery.data.data.is_short_selling_strategy
        ? "Short"
        : "Long";
    }
  };

  const getOpenTradeCond = () => {
    if (backtestQuery.data === undefined || backtestQuery.data === null) {
      return;
    }

    return backtestQuery.data?.data.open_trade_cond;
  };

  const getCloseTradeCond = () => {
    if (backtestQuery.data === undefined || backtestQuery.data === null) {
      return;
    }
    return backtestQuery.data?.data.close_trade_cond;
  };

  return (
    <>
      <div style={{ marginTop: "16px" }}>
        <ChakraCard heading={<Heading size="md">Trading criteria</Heading>}>
          <pre style={{ color: COLOR_CONTENT_PRIMARY }}>
            Strategy type: {getStrategyType()}
          </pre>
          <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
            {getOpenTradeCond()}
          </pre>
          <pre style={{ marginTop: "8px", color: COLOR_CONTENT_PRIMARY }}>
            {getCloseTradeCond()}
          </pre>
        </ChakraCard>
      </div>

      <div style={{ marginTop: "16px" }}>
        <ChakraCard heading={<Heading size="md">Strategy</Heading>}>
          <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
            Use time based close:{" "}
            {backtestQuery.data.data.use_time_based_close
              ? `True, ${backtestQuery.data.data.klines_until_close} candles`
              : "False"}
          </pre>
          <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
            Use stop loss based close:{" "}
            {backtestQuery.data.data.use_stop_loss_based_close
              ? `True, ${backtestQuery.data.data.stop_loss_threshold_perc}%`
              : "False"}
          </pre>
          <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
            Use profit (%) based close:{" "}
            {backtestQuery.data.data.use_profit_based_close
              ? `True, ${backtestQuery.data.data.take_profit_threshold_perc}%`
              : "False"}
          </pre>
        </ChakraCard>
      </div>

      {modelQuery && (
        <div style={{ marginTop: "16px" }}>
          <ChakraCard heading={<Heading size="md">Model</Heading>}>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              {backtestQuery.data.data.ml_enter_long_cond}
            </pre>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              {backtestQuery.data.data.ml_exit_long_cond}
            </pre>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              {backtestQuery.data.data.ml_enter_short_cond}
            </pre>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              {backtestQuery.data.data.ml_exit_short_cond}
            </pre>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "32px" }}>
              Epoch nr: {backtestQuery.data.data.model_train_epoch}
            </pre>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              {modelQuery.data?.model_code}
            </pre>
            <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
              {modelQuery.data?.optimizer_and_criterion_code}
            </pre>
          </ChakraCard>
        </div>
      )}
    </>
  );
};
