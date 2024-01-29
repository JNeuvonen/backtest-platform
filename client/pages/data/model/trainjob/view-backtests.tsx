import React from "react";
import { useTrainJobBacktests } from "../../../../clients/queries/queries";
import { usePathParams } from "../../../../hooks/usePathParams";
import { RowItem, SmallTable } from "../../../../components/tables/Small";
import { Spinner } from "@chakra-ui/react";
import { WithLabel } from "../../../../components/form/WithLabel";
import { BacktestObject } from "../../../../clients/queries/response-types";
import { Link } from "react-router-dom";

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

  const getBacktestsRows = (data: BacktestObject[]) => {
    const ret = [] as RowItem[];
    data.sort((a, b) => {
      return b.end_balance - a.end_balance;
    });
    data.forEach((item, i) => {
      ret.push([
        <Link className="link-default" to="/" key={i}>
          {i}
        </Link>,
        item.start_balance,
        item.end_balance,
      ]);
    });
    return ret;
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
