import React from "react";
import { useTrainJobBacktests } from "../../../../clients/queries/queries";
import { usePathParams } from "../../../../hooks/usePathParams";
import { SmallTable } from "../../../../components/tables/Small";
import { Spinner } from "@chakra-ui/react";
import { WithLabel } from "../../../../components/form/WithLabel";
import { BacktestObject } from "../../../../clients/queries/response-types";

const BACKTEST_TABLE_COLUMNS = ["Id", "Start balance", "End balance"];

export const BacktestsPage = () => {
  const { trainJobId } = usePathParams<{
    trainJobId: string;
    datasetName?: string;
  }>();
  const { data } = useTrainJobBacktests(trainJobId);

  if (!data) {
    return <Spinner />;
  }

  const getBacktestsRows = (data: BacktestObject) => {
    return [];
  };

  return (
    <div>
      <WithLabel label="Backtests">
        <SmallTable
          rows={getBacktestsRows(data)}
          columns={BACKTEST_TABLE_COLUMNS}
        ></SmallTable>
      </WithLabel>
    </div>
  );
};
