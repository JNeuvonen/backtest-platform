import React from "react";
import Title from "../../components/Title";
import { usePathParams } from "../../hooks/usePathParams";
import { useColumnQuery } from "../../clients/queries/queries";
import { SmallTable } from "../../components/tables/Small";
import { Spinner } from "@chakra-ui/react";
import { Column } from "../../clients/queries/response-types";
import { makeUniDimensionalTableRows } from "../../utils/table";
import { roundNumberDropRemaining } from "../../utils/number";

interface RouteParams {
  datasetName: string;
  columnName: string;
}

interface ColumnsRes {
  timeseries_col: string | null;
  column: Column;
}

const COLUMNS_STATS_TABLE: string[] = [
  "Timeseries column",
  "Nulls",
  "Nr rows",
  "Max value",
  "Mean",
  "Median",
  "Min",
  "Std dev",
];

export const DatasetColumnInfoPage = () => {
  const { datasetName, columnName } = usePathParams<RouteParams>();
  const { data, isLoading, isFetching, refetch } = useColumnQuery(
    datasetName,
    columnName
  );

  if (!data?.res) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const res = data.res;

  const getStatsRows = (data: ColumnsRes) => {
    const rows: (string | number)[] = [];

    if (data.timeseries_col) {
      rows.push(data.timeseries_col);
    }
    rows.push(roundNumberDropRemaining(data.column.null_count, 3));
    rows.push(data.column.rows.length);
    rows.push(roundNumberDropRemaining(data.column.stats.max, 3));
    rows.push(roundNumberDropRemaining(data.column.stats.mean, 3));
    rows.push(roundNumberDropRemaining(data.column.stats.median, 3));
    rows.push(roundNumberDropRemaining(data.column.stats.min, 3));
    rows.push(roundNumberDropRemaining(data.column.stats.std_dev, 3));
    return makeUniDimensionalTableRows(rows);
  };

  return (
    <div>
      <Title>Column {columnName}</Title>
      <SmallTable
        columns={COLUMNS_STATS_TABLE}
        rows={getStatsRows(res)}
        containerStyles={{ maxWidth: "800px", marginTop: "16px" }}
      />
    </div>
  );
};
