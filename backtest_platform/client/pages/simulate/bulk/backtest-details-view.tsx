import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import { useBacktestById } from "../../../clients/queries/queries";

interface PathParams {
  massPairTradeBacktestId: number;
}

export const LongShortBacktestsDetailsView = () => {
  const { massPairTradeBacktestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(massPairTradeBacktestId));

  console.log(backtestQuery.data);
  return <div></div>;
};
