import React from "react";
import { ChakraCard } from "../../../../components/chakra/Card";
import { Heading } from "@chakra-ui/react";
import { COLOR_CONTENT_PRIMARY } from "../../../../utils/colors";
import { FetchBacktestByIdRes } from "../../../../clients/queries/response-types";
import { UseQueryResult } from "@tanstack/react-query";

interface Props {
  backtestQuery: UseQueryResult<FetchBacktestByIdRes | null, unknown>;
}
export const TradingCriteriaCard = (props: Props) => {
  const { backtestQuery } = props;
  if (backtestQuery.data === undefined || backtestQuery.data === null) {
    return;
  }
  return (
    <>
      <div style={{ marginTop: "16px" }}>
        <ChakraCard heading={<Heading size="md">Trading criteria</Heading>}>
          <pre style={{ color: COLOR_CONTENT_PRIMARY }}>
            Strategy type:{" "}
            {backtestQuery.data.data.is_short_selling_strategy
              ? "Short"
              : "Long"}
          </pre>
          <pre style={{ color: COLOR_CONTENT_PRIMARY, marginTop: "16px" }}>
            {backtestQuery.data.data.open_trade_cond}
          </pre>
          <pre style={{ marginTop: "8px", color: COLOR_CONTENT_PRIMARY }}>
            {backtestQuery.data.data.close_trade_cond}
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
    </>
  );
};
