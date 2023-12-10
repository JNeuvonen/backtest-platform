import React from "react";
import { Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { getDateStr } from "../../utils/date";

interface LineItem {
  stroke: string;
  dataKey: string;
}

interface Props {
  data: Object[];
  xAxisDataKey: string;
  lines: LineItem[];
}

export const ColumnChart = ({ data, xAxisDataKey, lines }: Props) => {
  return (
    <LineChart width={1000} height={300} data={data}>
      <XAxis dataKey={xAxisDataKey} tickFormatter={getDateStr} />
      <YAxis />
      <Tooltip
        labelFormatter={getDateStr}
        contentStyle={{
          color: "white",
          background: "black",
          border: "1px solid black",
        }}
      />
      <Legend />

      {lines.map((item, i: number) => {
        return (
          <Line
            key={i}
            type="monotone"
            dataKey={item.dataKey}
            stroke={item.stroke}
            dot={false}
          />
        );
      })}
    </LineChart>
  );
};
