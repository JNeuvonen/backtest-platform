import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useManyBacktests,
  useMassbacktest,
} from "../../../clients/queries/queries";
import { Spinner } from "@chakra-ui/react";
import { BacktestObject } from "../../../clients/queries/response-types";

interface PathParams {
  massBacktestId: number;
}

const getMassSimEquityCurvesData = (backtests: BacktestObject[]) => {
  const ret = [];

  return [];
};

export const InvidualMassbacktestDetailsPage = () => {
  const { massBacktestId } = usePathParams<PathParams>();
  const massBacktestQuery = useMassbacktest(Number(massBacktestId));
  const useManyBacktestsQuery = useManyBacktests(
    massBacktestQuery.data?.backtest_ids || []
  );

  if (
    massBacktestQuery.isLoading ||
    !massBacktestQuery.data ||
    useManyBacktestsQuery.isLoading ||
    !useManyBacktestsQuery.data
  ) {
    return <Spinner />;
  }

  console.log(massBacktestQuery.data, useManyBacktestsQuery.data);

  return <div>Hello world</div>;
};
