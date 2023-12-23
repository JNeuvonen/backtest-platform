import React from "react";
import { useColumnQuery } from "../../clients/queries/queries";
import { ColumnChart } from "../../components/charts/column";
import { Spinner } from "@chakra-ui/react";

interface ColumnModalContentProps {
  datasetName: string;
  columnName: string;
}

export const ColumnModal = ({
  columnName,
  datasetName,
}: ColumnModalContentProps) => {
  const { data, isLoading } = useColumnQuery(datasetName, columnName);

  const massageDataForChart = (
    rows: number[][],
    kline_open_time: number[][]
  ) => {
    const itemCount = rows.length;
    const skipItems = Math.max(1, Math.floor(itemCount / 1000));
    const ret: Object[] = [];

    for (let i = 0; i < itemCount; i++) {
      if (i % skipItems === 0) {
        const item = rows[i];
        const rowObject = {};
        rowObject[columnName] = item[0];
        rowObject["kline_open_time"] = kline_open_time[i][0];
        ret.push(rowObject);
      }
    }

    return ret;
  };

  if (isLoading) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const rows = data?.res.column.rows;
  const kline_open_time = data?.res.column.kline_open_time;
  if (!rows || !kline_open_time) return null;
  return (
    <div>
      <ColumnChart
        data={massageDataForChart(rows, kline_open_time)}
        xAxisDataKey={"kline_open_time"}
        lines={[{ dataKey: columnName, stroke: "red" }]}
      />
    </div>
  );
};
